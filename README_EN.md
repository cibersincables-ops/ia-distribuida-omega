# 🧠 Distributed AI · Public Aqueduct of Knowledge

**Current Status:** Phase 1 — Middleware Proxy Ω (Phase 0 validated: AUC 0.9539)

Intelligence should not be a privilege of centralized servers, but an emergent property
of the network. This project builds a collective infrastructure where thousands of
volunteer nodes execute fragmented inference in a private, verifiable, and manipulation-resistant way.

## 🏗️ Architecture (Phase 1 in Development)

The project has evolved to integrate with **Exo** as the distributed inference engine,
adding a semantic verification layer and a sovereign network infrastructure.

### 🔧 Inference Engine: Exo
[Exo](https://github.com/exo-explore/exo) handles automatic model sharding
across heterogeneous devices (Mac, Linux, iOS). It provides the computational muscle
without centralized servers.

### 🛡️ Immune System: Middleware Proxy Ω
A semantic customs layer built on top of Exo:
- **Real-time Validation:** every node response is analyzed by Ω
  (experimentally validated AUC 0.9539) before reaching the user.
- **FIX Protocol:** messages are packaged under the financial messaging standard
  for full traceability and auditability of each fragment.
- **2/3 Quorum:** triple redundancy with majority voting to discard
  malicious nodes or hardware errors.

### 🌐 ISP-Grade Network Infrastructure
- **VPN Mesh:** Tailscale/WireGuard for secure tunnels between volunteer nodes.
- **ACO Routing:** ant colony algorithms that prioritize routes through nodes
  with higher honesty history.
- **Hardware Ready:** designed for MikroTik/Cisco equipment and adapted OSPF protocols.

## 📊 Current Status

| Phase | Status | Key Metric |
| :--- | :--- | :--- |
| **Phase 0:** Ω Validation | ✅ Complete | AUC-ROC 0.9539 |
| **Phase 1:** Middleware Proxy | 🔴 In Progress | Integration with Exo + FIX |
| **Phase 2:** Scaling | ⚪ Planned | Latency p95 < 200ms |
| **Phase 3:** Open Network | ⚪ Planned | >500 volunteer nodes |

## 🧪 Phase 0: The Experiment

The Ω Formula was validated with sentence-transformers embeddings over 1,000 text pairs.
- **AUC-ROC:** 0.9539
- **Coherent mean Ω:** 0.433
- **Divergent mean Ω:** 0.064
- **Separation:** 0.369

## 📖 Full Documentation

👉 **[Open complete architecture](docs/arquitectura.html)** — Tech stack, sequence diagrams,
FIX protocol, privacy layers, roadmap (currently in Spanish).

## 🤝 Contributing

The project is entering Phase 1. The most valuable contributions right now are:
- **Middleware Proxy:** building the Ω validation layer over Exo.
- **FIX Integration:** implementing the financial protocol for fragment traceability.
- **ACO Routing:** simulating and testing ant colony algorithms on OSPF/BGP.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) and issues tagged `good first issue`.

## 📄 License

[MIT](LICENSE) — Knowledge is public infrastructure.