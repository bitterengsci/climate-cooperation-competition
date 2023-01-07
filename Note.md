

* [``rice.py``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py) includes interactions between the agents and the environment. **[``rice.py``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py) is the main script to be modified.**

* [``rice_helpers.py``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice_helpers.py) includes all the socioeconomic and climate dynamics. [``rice_helpers.py``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice_helpers.py) should not be changed.

<!-- TOC -->

- [1. Add extra observations](#1-add-extra-observations)
- [2. Add actions](#2-add-actions)
- [3. Implement the logic for negotiation protocols](#3-implement-the-logic-for-negotiation-protocols)
- [4. Masking](#4-masking)
- [5. Implement and/or modify the logic of action masking?](#5-implement-andor-modify-the-logic-of-action-masking)

<!-- /TOC -->


# 1. Add extra observations

To add extra observations or make changes to the observation space, at least two functions must be modified.
1.   [`generate_observation()`](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L379)
2.   [`reset()`](https://github.com/mila-iqia/climate-cooperation-competition/blob/)

As an example, [here](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L428) are the features added when the naive bilateral negotiation protocol is enabled in the simulator: 

```python
if self.negotiation_on:
    global_features += ["stage"]

    public_features += []

    private_features += [
        "minimum_mitigation_rate_all_regions",
    ]

    bilateral_features += [
        "promised_mitigation_rate",
        "requested_mitigation_rate",
        "proposal_decisions",
    ]

shared_features = np.array([])
for feature in global_features + public_features:
    shared_features = np.append(
        shared_features,
        self.flatten_array(
            self.global_state[feature]["value"][self.timestep]
            / self.global_state[feature]["norm"]
        ),
    )
```

# 2. Add actions

By default, agents' actions are contained in [`self.actions_nvec`](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L136) during [`init()`](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L64):

```python
self.actions_nvec = (
    self.savings_action_nvec
    + self.mitigation_rate_action_nvec
    + self.export_action_nvec
    + self.import_actions_nvec
    + self.tariff_actions_nvec
)
```

Extra actions related to the negotiation protocol can be appended to `self.actions_nvec`.
It is important that extra actions be appended at the **end** of `self.actions_nvec`.
``` python 
# Each region proposes to each other region
# self mitigation and their mitigation values
self.proposal_actions_nvec = (
    [self.num_discrete_action_levels] * 2 * self.num_regions
)

# Each region evaluates a proposal from every other region,
# either accept or reject.
self.evaluation_actions_nvec = [2] * self.num_regions

# extra actions are appended to the end of self.actions_nvec
self.actions_nvec += (
    self.proposal_actions_nvec + self.evaluation_actions_nvec
)

```

# 3. Implement the logic for negotiation protocols

The baseline logic for bilateral negotiation actions is a naive bargain process with two steps:
1. A [``proposal_step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L536) for each agent to propose certains actions to other agents, for example a minimum mitigation rate.
2. An [``evaluation_step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L585) for each agent to evaluation other agents' proposals. 

These functions describe how the negotiations actions affect the observation space and the action masking (for more, see the next section).
Both steps are done sequentially in the [``step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L346) function in [``rice.py``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py): 

```python
if self.negotiation_on:
    # Note: The '+1` below is for the climate_and_economy_simulation_step
    self.stage = self.timestep % (self.num_negotiation_stages + 1)
    self.set_global_state(
        "stage", self.stage, self.timestep, dtype=self.int_dtype
    )
    if self.stage == 1:
        return self.proposal_step(actions)

    if self.stage == 2:
        return self.evaluation_step(actions)

return self.climate_and_economy_simulation_step(actions)
```
Once the stages of the negotiation protocol are concluded, then the [`climate_and_economy_simulation_step()`](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L651) implements the socioeconomic and climate dynamics associated with the updated observation space and masked actions.

We expect competitors to propose different mechanisms to encourage global cooperation along climate and economic objectives.
Participants should therefore modify this code to match the logic of their proposed negotiation protocol, even proposing new functions to replace [``proposal_step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L536), [``evaluation_step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L585) and the code above.

For example, competitors could propose a mechanism to form [dynamic climate clubs](https://williamnordhaus.com/publications/climate-clubs-overcoming-free-riding-international-climate-policy), where admittance is based on a minimum mitigation rate. Club members enjoy lower tariffs when trading with other club members, while non-members, who do not have to contribute to mitigation, suffer heavy tariffs when trading with club members.

# 4. Masking

Action masking determines the feasible subspace of the action space according to the negotiation protocol. Action masks are set before agents choose their actions, so the agent explicitly chooses from the feasible action subspace.
To implement this logic, actions masks are modified in the [``evaluation_step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L585), after the [``proposal_step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L536) and [``evaluation_step()``](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L585), but before the [`climate_and_economy_simulation_step()`](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L651). This way, the regions are prohibited from taking actions outside of the feasible action subspace.

For example, during the bilateral negotiation process, regions that agree to implement minimum mitigation rates are required to do so. 

```python
for region_id in range(self.num_regions):
    outgoing_accepted_mitigation_rates = [
        self.global_state["promised_mitigation_rate"]["value"][
            self.timestep, region_id, j
        ]
        * self.global_state["proposal_decisions"]["value"][
            self.timestep, j, region_id
        ]
        for j in range(self.num_regions)
    ]
    incoming_accepted_mitigation_rates = [
        self.global_state["requested_mitigation_rate"]["value"][
            self.timestep, j, region_id
        ]
        * self.global_state["proposal_decisions"]["value"][
            self.timestep, region_id, j
        ]
        for j in range(self.num_regions)
    ]

    self.global_state["minimum_mitigation_rate_all_regions"]["value"][
        self.timestep, region_id
    ] = max(
        outgoing_accepted_mitigation_rates + incoming_accepted_mitigation_rates
    )
```

# 5. Implement and/or modify the logic of action masking?

The logic behind action masks is implemented in [`generate_action_mask()`](https://github.com/mila-iqia/climate-cooperation-competition/blob/main/rice.py#L506).
`mask_dict` gives the mapping for each region to its corresponding action `mask`. In the current implementation, `mask` is a binary vector where `0` indicates an action that is not allowed, and `1` indicates an action that is allowed.

For example, in the bilateral negotiation protocol, the action mask is based on the minimum mitigation rate for each region (see code below).
```python
def generate_action_mask(self):
    '''
    Generate action masks.
    '''
    mask_dict = {region_id: None for region_id in range(self.num_regions)}
    for region_id in range(self.num_regions):
        mask = self.default_agent_action_mask.copy()
        if self.negotiation_on:
            minimum_mitigation_rate = int(round(
                self.global_state["minimum_mitigation_rate_all_regions"]["value"][
                    self.timestep, region_id
                ]
                * self.num_discrete_action_levels
            ))
            mitigation_mask = np.array(
                [0 for _ in range(minimum_mitigation_rate)]
                + [
                    1
                    for _ in range(
                        self.num_discrete_action_levels - minimum_mitigation_rate
                    )
                ]
            )
            mask_start = sum(self.savings_action_nvec)
            mask_end = mask_start + sum(self.mitigation_rate_action_nvec)
            mask[mask_start:mask_end] = mitigation_mask
        mask_dict[region_id] = mask

    return mask_dict
```

