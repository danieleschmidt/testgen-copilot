# 🧭 Project Vision

> A short 2–3 sentence description of what this repo does, for whom, and why.

---

# 📅 12-Week Roadmap

## Increment I1
- **Themes**: Security, Developer UX
- **Goals / Epics**
  - Harden CLI input handling
  - Stabilize CI pipeline
  - Improve VS Code extension usability
- **Definition of Done**
  - All inputs validated and unit tested
  - CI passes reliably on main and PRs
  - Extension docs published and commands tested

## Increment I2
- **Themes**: Performance, Quality
- **Goals / Epics**
  - Optimize test generation for large projects
  - Enhance test quality scoring accuracy
  - Parallelize coverage analysis
- **Definition of Done**
  - Benchmarks show >20% speedup on repo scan
  - Quality scorer handles parameterized tests
  - Coverage analyzer runs concurrently without race conditions

## Increment I3
- **Themes**: Observability, Community
- **Goals / Epics**
  - Add structured logging and metrics
  - Publish contribution guidelines and examples
  - Prepare first stable release
- **Definition of Done**
  - Logs emit JSON and integrate with existing tools
  - CONTRIBUTING updated with sample workflows
  - v1 release notes finalized in CHANGELOG

---

# 👌 Epic & Task Checklist

### 🔒 Increment 1: Security & Refactoring
- [x] **EPIC** Harden CLI input handling
  - [x] Validate paths and sanitize user-supplied options
  - [x] Add schema validation for config files
- [x] **EPIC** Improve CI stability
  - [x] Replace flaky integration tests
  - [x] Enable parallel test execution
- [x] **EPIC** Extension usability improvements
  - [x] Document commands in README
  - [x] Publish usage examples

### 🚀 Increment 2: Performance & Quality
- [ ] **EPIC** Speed up project scan
  - [ ] Profile generator on large repos
  - [ ] Implement caching layer
- [ ] **EPIC** Enhance quality scoring
  - [ ] Support parametric tests
  - [ ] Flag missing fixtures
- [ ] **EPIC** Concurrent coverage analyzer
  - [ ] Use multiprocessing for file scans
  - [ ] Provide progress output

### 📊 Increment 3: Observability & Release
- [ ] **EPIC** Structured logging & metrics
  - [ ] Emit JSON logs from CLI
  - [ ] Track generation time and security issues found
- [ ] **EPIC** Community onboarding
  - [ ] Expand CONTRIBUTING guide
  - [ ] Provide issue templates
- [ ] **EPIC** First stable release
  - [ ] Finalize CHANGELOG
  - [ ] Tag v1.0.0

---

# ⚠️ Risks & Mitigation
- Reliance on external LLM APIs may limit offline usage → provide fallback mock generator
- CI environment differences could cause flaky tests → use containerized runners
- Rapid dependency updates might break CLI → pin versions and run nightly tests
- Large projects could exhaust memory during analysis → implement streaming and batching
- New contributors may face steep learning curve → supply quick-start scripts and docs

---

# 📊 KPIs & Metrics
- [ ] ≥85% test coverage
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core service
- [ ] 100% secrets loaded from vault/env

---

# 👥 Ownership & Roles
- **DevOps**: maintain CI/CD and infrastructure
- **Backend**: own CLI, generator, and analysis modules
- **Frontend**: maintain VS Code extension
- **QA**: enforce test quality and coverage targets
