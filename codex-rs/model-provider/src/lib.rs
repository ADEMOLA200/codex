//! Runtime model provider resolution.
//!
//! `codex_model_provider_info` owns the config-facing provider metadata. This
//! crate turns that metadata into the narrow runtime facade that model-facing
//! callsites should depend on. The first slice is intentionally auth-only; the
//! facade can grow transport, catalog, and capability accessors as those
//! callsites move behind provider ownership.

use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::config_types::ModelProviderAuthInfo;
use std::error::Error;
use std::fmt;

/// Stable identifier for a configured model provider.
pub type ModelProviderId = String;

/// Runtime facade for provider-owned model behavior.
///
/// This trait starts with only auth-facing accessors. Add model listing,
/// Responses client construction, and optional specialized clients here as
/// those callsites move behind provider ownership.
pub trait ModelProvider {
    fn id(&self) -> &str;
    fn info(&self) -> &ModelProviderInfo;
    fn auth_provider(&self) -> &ProviderAuth;
}

/// Auth strategy selected for a resolved model provider.
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderAuth {
    /// OpenAI-managed auth through API key, ChatGPT, or ChatGPT auth tokens.
    OpenAi,
    /// Bearer token read from an environment variable.
    EnvBearer {
        env_key: String,
        env_key_instructions: Option<String>,
    },
    /// Bearer token embedded directly in provider config.
    ExperimentalBearer { token: String },
    /// Bearer token produced by an external command.
    ExternalBearer { config: ModelProviderAuthInfo },
    /// No provider auth should be attached.
    None,
}

impl ProviderAuth {
    /// Whether this auth strategy uses OpenAI account/API-key auth flows.
    pub fn requires_openai_auth(&self) -> bool {
        matches!(self, Self::OpenAi)
    }
}

/// Resolved runtime provider facade.
///
/// This type starts as an auth-only facade. Future provider-owned behavior
/// should be added as methods/fields on this type rather than by teaching
/// callsites to branch on provider IDs.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModelProvider {
    id: ModelProviderId,
    info: ModelProviderInfo,
    auth: ProviderAuth,
}

impl ResolvedModelProvider {
    /// Resolve config-facing provider metadata into the runtime provider facade.
    pub fn resolve(
        id: impl Into<ModelProviderId>,
        info: ModelProviderInfo,
    ) -> Result<Self, ResolveProviderError> {
        info.validate()
            .map_err(ResolveProviderError::InvalidConfig)?;
        let auth = resolve_auth(&info)?;
        Ok(Self {
            id: id.into(),
            info,
            auth,
        })
    }

    pub fn id(&self) -> &str {
        self.id.as_str()
    }

    pub fn info(&self) -> &ModelProviderInfo {
        &self.info
    }

    /// Return the provider-owned auth strategy.
    pub fn auth_provider(&self) -> &ProviderAuth {
        &self.auth
    }
}

impl ModelProvider for ResolvedModelProvider {
    fn id(&self) -> &str {
        self.id.as_str()
    }

    fn info(&self) -> &ModelProviderInfo {
        &self.info
    }

    fn auth_provider(&self) -> &ProviderAuth {
        &self.auth
    }
}

fn resolve_auth(info: &ModelProviderInfo) -> Result<ProviderAuth, ResolveProviderError> {
    match (
        info.requires_openai_auth,
        info.env_key.as_ref(),
        info.experimental_bearer_token.as_ref(),
        info.auth.as_ref(),
    ) {
        (true, None, None, None) => Ok(ProviderAuth::OpenAi),
        (false, Some(env_key), None, None) => Ok(ProviderAuth::EnvBearer {
            env_key: env_key.clone(),
            env_key_instructions: info.env_key_instructions.clone(),
        }),
        (false, None, Some(token), None) => Ok(ProviderAuth::ExperimentalBearer {
            token: token.clone(),
        }),
        (false, None, None, Some(config)) => Ok(ProviderAuth::ExternalBearer {
            config: config.clone(),
        }),
        (false, None, None, None) => Ok(ProviderAuth::None),
        _ => Err(ResolveProviderError::InvalidConfig(
            "provider auth settings are ambiguous".to_string(),
        )),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveProviderError {
    InvalidConfig(String),
}

impl fmt::Display for ResolveProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid provider config: {message}"),
        }
    }
}

impl Error for ResolveProviderError {}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_model_provider_info::ModelProviderInfo;
    use codex_model_provider_info::WireApi;
    use codex_utils_absolute_path::AbsolutePathBuf;
    use pretty_assertions::assert_eq;
    use std::num::NonZeroU64;

    fn provider() -> ModelProviderInfo {
        ModelProviderInfo {
            name: "Test Provider".to_string(),
            base_url: Some("https://example.com/v1".to_string()),
            env_key: None,
            env_key_instructions: None,
            experimental_bearer_token: None,
            auth: None,
            wire_api: WireApi::Responses,
            query_params: None,
            http_headers: None,
            env_http_headers: None,
            request_max_retries: None,
            stream_max_retries: None,
            stream_idle_timeout_ms: None,
            websocket_connect_timeout_ms: None,
            requires_openai_auth: false,
            supports_websockets: false,
        }
    }

    #[test]
    fn resolves_openai_auth() {
        let info = ModelProviderInfo::create_openai_provider(/*base_url*/ None);

        let provider = ResolvedModelProvider::resolve("openai", info.clone()).unwrap();

        assert_eq!(provider.id(), "openai");
        assert_eq!(provider.info(), &info);
        assert_eq!(provider.auth_provider(), &ProviderAuth::OpenAi);
        assert!(provider.auth_provider().requires_openai_auth());
    }

    #[test]
    fn resolved_provider_implements_model_provider_facade() {
        fn auth_from_provider(provider: &impl ModelProvider) -> &ProviderAuth {
            provider.auth_provider()
        }

        let provider = ResolvedModelProvider::resolve(
            "openai",
            ModelProviderInfo::create_openai_provider(/*base_url*/ None),
        )
        .unwrap();

        assert_eq!(auth_from_provider(&provider), &ProviderAuth::OpenAi);
    }

    #[test]
    fn resolves_env_bearer_auth() {
        let mut info = provider();
        info.env_key = Some("TEST_API_KEY".to_string());
        info.env_key_instructions = Some("Set TEST_API_KEY.".to_string());

        let provider = ResolvedModelProvider::resolve("custom", info).unwrap();

        assert_eq!(
            provider.auth_provider(),
            &ProviderAuth::EnvBearer {
                env_key: "TEST_API_KEY".to_string(),
                env_key_instructions: Some("Set TEST_API_KEY.".to_string()),
            }
        );
    }

    #[test]
    fn resolves_experimental_bearer_auth() {
        let mut info = provider();
        info.experimental_bearer_token = Some("token".to_string());

        let provider = ResolvedModelProvider::resolve("custom", info).unwrap();

        assert_eq!(
            provider.auth_provider(),
            &ProviderAuth::ExperimentalBearer {
                token: "token".to_string(),
            }
        );
    }

    #[test]
    fn resolves_external_bearer_auth() {
        let mut info = provider();
        let auth_config = ModelProviderAuthInfo {
            command: "credential-helper".to_string(),
            args: vec!["token".to_string()],
            timeout_ms: NonZeroU64::new(10_000).unwrap(),
            refresh_interval_ms: 300_000,
            cwd: AbsolutePathBuf::from_absolute_path("/tmp").unwrap(),
        };
        info.auth = Some(auth_config.clone());

        let provider = ResolvedModelProvider::resolve("custom", info).unwrap();

        assert_eq!(
            provider.auth_provider(),
            &ProviderAuth::ExternalBearer {
                config: auth_config,
            }
        );
    }

    #[test]
    fn resolves_no_auth_for_custom_provider() {
        let provider = ResolvedModelProvider::resolve("custom", provider()).unwrap();

        assert_eq!(provider.auth_provider(), &ProviderAuth::None);
        assert!(!provider.auth_provider().requires_openai_auth());
    }

    #[test]
    fn rejects_invalid_command_auth() {
        let mut info = provider();
        info.auth = Some(ModelProviderAuthInfo {
            command: " ".to_string(),
            args: Vec::new(),
            timeout_ms: NonZeroU64::new(10_000).unwrap(),
            refresh_interval_ms: 300_000,
            cwd: AbsolutePathBuf::from_absolute_path("/tmp").unwrap(),
        });

        let err = ResolvedModelProvider::resolve("custom", info).unwrap_err();

        assert_eq!(
            err,
            ResolveProviderError::InvalidConfig(
                "provider auth.command must not be empty".to_string()
            )
        );
    }
}
