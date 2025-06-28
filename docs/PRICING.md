# Custom Model Pricing

Polyglot AI uses a comprehensive, cached list of model prices from the LiteLLM project. However, you can provide your own pricing file to override these defaults or add pricing for models that are not on the public list (e.g., private, local, or newly released models).

This feature allows for precise cost tracking in any environment.

## How to Enable Custom Pricing

You can specify a custom pricing file in two ways. The file can be in **YAML** (recommended) or **TOML** format.

### 1. In `polyglot.toml` (Recommended)

Add the `custom-pricing-file` key to the top level of your `polyglot.toml` configuration file. The path can be absolute or relative to the location of `polyglot.toml`.

```toml
# polyglot.toml - using a YAML file
custom-pricing-file = "custom_prices.yaml"

# Or using a TOML file
# custom-pricing-file = "custom_prices.toml"

[[endpoints]]
# ... your endpoint definitions
```

### 2. Using a CLI Flag

You can specify the path to a pricing file at runtime using the `--custom-pricing-file` flag. This flag will **override** any path set in `polyglot.toml`.

```bash
pai --chat --custom-pricing-file /path/to/my/prices.yaml
```

## Custom Pricing File Format

The custom pricing file defines pricing for specific `(provider, model)` combinations. The provider name must match the `name` of an endpoint defined in your `[[endpoints]]` table, and the model name must match the model string you use (e.g., via `--model` or in an Arena config).

**All costs are specified in USD per 1 million tokens.**

The top-level key must be `pricing`.

---

## YAML Format Examples (Recommended)

This is the recommended format due to its readability for nested data structures.

### Example 1: Simple Flat-Rate Override (YAML)

This example overrides the pricing for OpenAI's `gpt-4o-mini` and provides pricing for a local model running on an endpoint named `local-lm-studio`.

```yaml
# custom_prices.yaml
pricing:
  openai:
    gpt-4o-mini:
      # This will override the default price for gpt-4o-mini.
      input_cost: 0.20   # $0.20 / 1M input tokens
      output_cost: 0.80  # $0.80 / 1M output tokens

  # Provide pricing for a local model.
  local-lm-studio:
    "Meta-Llama-3-8B-Instruct":
      # Local models have no cost, so we set it to 0.
      input_cost: 0.0
      output_cost: 0.0
```

### Example 2: Tiered Pricing (YAML)

`up_to` defines the upper bound for a tier. Use `-1` to represent infinity for the final tier.

```yaml
pricing:
  anthropic:
    "claude-3-5-sonnet-20240620":
      # If tiers are present, they are used instead of these flat rates.
      input_cost: 3.00
      output_cost: 15.00
      
      # Tiers for input tokens
      input_tiers:
        - up_to: 1000000
          cost: 2.50
        - up_to: -1 # Applies to all tokens after the previous tier
          cost: 2.00
          
      # Tiers for output tokens
      output_tiers:
        - up_to: 1000000
          cost: 12.00
        - up_to: -1
          cost: 10.00
```

### Example 3: Time-Based Pricing (YAML)

`start_hour` and `end_hour` are inclusive and based on a 24-hour clock (0-23) in UTC. If no window matches, the system falls back to any "anytime" rates defined.

```yaml
pricing:
  some_provider:
    some_model:
      # "Anytime" / Fallback rates
      input_cost: 10.00
      output_cost: 20.00

      # A list of time-specific pricing windows.
      time_windows:
        # Peak hours (9 AM to 5 PM UTC)
        - start_hour: 9
          end_hour: 17
          input_cost: 15.00
          output_cost: 25.00
        # Off-peak hours (10 PM to 6 AM UTC) - an overnight window
        - start_hour: 22
          end_hour: 6
          input_cost: 5.0
          output_cost: 10.0
```

### Example 4: Complex Combination (YAML)

```yaml
pricing:
  another_provider:
    super-model-v9:
      # Fallback "anytime" flat rate
      input_cost: 5.0
      output_cost: 15.0

      # Fallback "anytime" tiered rates (used if no time window matches)
      output_tiers:
        - up_to: 4096
          cost: 15.0
        - up_to: -1
          cost: 25.0
      
      time_windows:
        # Weekend special (UTC Saturday/Sunday)
        - start_hour: 0
          end_hour: 23
          input_cost: 2.0
          output_cost: 6.0
        # Weekday with tiered pricing
        - start_hour: 8
          end_hour: 18
          # A flat input cost for this window
          input_cost: 7.0
          # But a tiered output cost
          output_tiers:
            - up_to: 2048
              cost: 18.0
            - up_to: -1
              cost: 30.0
```
---

## TOML Format Examples

TOML is also supported for backwards compatibility and preference.

### Example 1: Simple Flat-Rate Override (TOML)

```toml
# custom_prices.toml
[pricing]
  [pricing.openai]
    [pricing.openai."gpt-4o-mini"]
    input_cost = 0.20
    output_cost = 0.80
  [pricing.local-lm-studio]
    [pricing.local-lm-studio."Meta-Llama-3-8B-Instruct"]
    input_cost = 0.0
    output_cost = 0.0
```

### Example 2: Tiered Pricing (TOML)

```toml
[pricing.anthropic.claude-3-5-sonnet-20240620]
input_cost = 3.00
output_cost = 15.00
input_tiers = [
    { up_to = 1_000_000, cost = 2.50 },
    { up_to = -1, cost = 2.00 },
]
output_tiers = [
    { up_to = 1_000_000, cost = 12.00 },
    { up_to = -1, cost = 10.00 },
]
```

### Example 3: Time-Based Pricing (TOML)

```toml
[pricing.some_provider.some_model]
input_cost = 10.00
output_cost = 20.00
time_windows = [
    { start_hour = 9, end_hour = 17, input_cost = 15.00, output_cost = 25.00 },
    { start_hour = 22, end_hour = 6, input_cost = 5.0, output_cost = 10.0 },
]
```
