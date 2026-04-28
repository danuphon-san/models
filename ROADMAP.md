# Roadmap

`ml4t-models` is a finance-native model library for empirical asset pricing, stochastic discount
factor learning, and portfolio construction.

The roadmap is intentionally narrow: add model families that fit the existing architecture, have
clear reader value, and can be tested against real ML4T workflows without turning the package into
an unfocused paper zoo.

## Current Baseline

The current release already covers four coherent families:

- latent factors
  - `PCAModel`
  - `RPPCAModel`
  - `IPCAModel`
  - `CAEModel`
- direct asset prediction
  - `SAEModel`
- stochastic discount factor
  - `StochasticDiscountFactorModel`
- portfolio learning
  - `LinearFeaturePortfolioModel`
  - `LSTMPortfolioModel`
  - `DeepPortfolioModel`

The public contracts are:

- `PersistentPanelBatch`
- `CrossSectionBatch`
- `PortfolioSequenceBatch`

and the main architectural split is explicit:

- structural estimation
- forecasting or mapping
- backtest and diagnostic handoff

## Prioritization Rules

We should add models only when they satisfy most of the following:

- they define a reusable model family, not just an application paper
- they fit the existing batch and result contracts, or justify one clean new contract
- they broaden the library in a meaningful way
- they can be validated against real ML4T data or case-study workflows
- they do not require a separate agent, RL, or execution stack

We should avoid:

- survey-driven implementation work
- one-off domain applications that do not generalize
- speculative agentic or multi-agent systems before the core finance stack is mature

## Phase 1: Finish The Core

These are the highest-priority non-model items.

### 1. Case Study Migration And Validation

- finish routing the latent-factor case studies through `ml4t-models`
- rerun all affected case studies end-to-end
- confirm registry persistence and downstream analysis paths
- keep using `ml4t-diagnostic` for cross-sectional IC instead of local metric code

### 2. Notebook Integration

- keep Chapter 14 and Chapter 17 teaching notebooks implementation-first
- show the raw paper-faithful model implementation in the notebooks
- show the corresponding `ml4t-models` usage as the production path

### 3. Release Hardening

- stabilize public naming
- keep docs and examples synchronized with the shipped API
- prefer reference workflows over adding more families prematurely

## Phase 2: Next Model Additions

These are the best additions after the current baseline is stable.

### A. `ProximateFactorModel`

Paper:

- Pelger-Xiong, *Interpretable Sparse Proximate Factors* (`WKLRFQGN`)

Why next:

- clean extension of the latent-factor family
- strong reader value through interpretability
- fits the current persistent-panel latent-factor path

Expected family:

- `ml4t.models.latent_factors.ProximateFactorModel`

Expected contract:

- `PersistentPanelBatch`

### B. `ImplementableEfficientFrontierModel`

Paper:

- Jensen et al., *Machine Learning and the Implementable Efficient Frontier* (`N4F346PN`)

Why next:

- broadens the portfolio side without overlapping end-to-end deep allocators
- gives the library a strong weight-native portfolio construction model that is not purely neural

Expected family:

- `ml4t.models.portfolio.ImplementableEfficientFrontierModel`

Expected contract:

- `PortfolioSequenceBatch`

### C. `EconomicTargetFactorModel`

Paper:

- Bryzgalova et al., *Asset-Pricing Factors with Economic Targets* (`DKARC7UQ`)

Why later than A/B:

- very relevant
- but slightly more specialized than proximate factors
- best added after the sparse-factor path is in place

Expected family:

- `ml4t.models.latent_factors.EconomicTargetFactorModel`

Expected contract:

- `PersistentPanelBatch`

## Phase 3: Structured Extensions

These are good fits, but they likely need contract or architecture extensions.

### `StateVaryingFactorModel`

Paper:

- Pelger-Xiong, *State-Varying Factor Models of Large Dimensions* (`46GITYAN`)

Why later:

- likely needs an explicit state-conditioned latent-factor contract
- should follow once the static and sparse latent-factor paths are solid

### `WeakFactorModel`

Paper:

- Giglio et al., *Test Assets and Weak Factors* (`TLGNDFRP`)

Why later:

- useful bridge between RP-PCA and more inference-aware factor construction
- less central than the proximate-factor addition

### `RecursivePortfolioMachines`

Paper:

- Fan et al., *Recursive Portfolio Machines* (`L7QRDDQU`)

Why later:

- promising portfolio family
- but newer and architecturally heavier than the implementable-frontier path

## Phase 4: New Modalities

These broaden the library beyond tabular or panel inputs and should be treated as separate
expansions, not quick add-ons.

### `ImpliedVolatilitySurfaceModel`

Motivating paper:

- *Deep Learning from Implied Volatility Surfaces*

Why it matters:

- we already have options chain and options-analytics data
- this is a real new modality, not just another cross-sectional tabular model

What it likely needs:

- a new options-surface batch contract
- image or tensor encoders for volatility surfaces
- asset-level prediction outputs

Suggested family:

- `ml4t.models.options`

### Not Prioritized

- price-trend image models
- agentic investing systems
- LLM portfolio agents
- multi-agent strategy discovery

These may be interesting, but they are not the right next step for this library.

## Explicit Non-Goals For Now

The following should stay out of the immediate roadmap:

- RL trading agents
- execution agents
- generic LLM wrappers
- market-making systems
- one-off application papers with no reusable contract

## Suggested GitHub Organization

If we want outside contributions, the cleanest setup is:

### Milestones

- `v0.1.x Stabilization`
- `v0.2 Latent Factor Extensions`
- `v0.3 Portfolio Construction Extensions`
- `v0.4 Options Surface Models`

### Labels

- `family:latent-factors`
- `family:portfolio`
- `family:stochastic-discount-factor`
- `family:options`
- `area:docs`
- `area:tests`
- `area:integration`
- `good first issue`
- `help wanted`
- `research`

### First Candidate Issues

- Add `ProximateFactorModel`
- Add `ImplementableEfficientFrontierModel`
- Add `EconomicTargetFactorModel`
- Design an options-surface batch contract
- Add a first end-to-end options-surface forecasting baseline

## Recommended Order

If we execute this roadmap strictly, the next steps should be:

1. finish case-study validation and end-to-end reruns
2. stabilize public launch materials and examples
3. implement `ProximateFactorModel`
4. implement `ImplementableEfficientFrontierModel`
5. implement `EconomicTargetFactorModel`
6. design the options-surface contract

That keeps the library coherent while still broadening it in meaningful ways.
