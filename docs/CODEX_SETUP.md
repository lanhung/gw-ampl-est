# Codex CLI setup

The Vultr host uses Codex CLI `0.144.1` from the authoritative repository
root `/root/work/lensing-4`.

The machine-level file `/root/.codex/config.toml` is intentionally not part
of this Git repository. It was configured with:

```toml
approval_policy = "on-request"
sandbox_mode = "workspace-write"

[sandbox_workspace_write]
network_access = true
writable_roots = ["/root/work/lensing-4"]

[shell_environment_policy]
inherit = "core"
```

The official OpenAI developer-documentation MCP endpoint is registered as
`openaiDeveloperDocs`. A new Codex session is required after configuration
changes so that both `AGENTS.md` and the MCP server are loaded afresh.

`codex doctor` confirms that the configuration loads with restricted
filesystem access, network enabled and `OnRequest` approval. At the time of
the Phase 0 check, the host could not reach one or more OpenAI provider
endpoints over HTTP and WebSocket. This does not affect the completed local
and AutoDL audit, but it may affect a newly launched Codex session until the
Vultr outbound-network condition is resolved.

Phase 0 used SSH only. It did not browse the web, install scientific
packages, or download datasets.
