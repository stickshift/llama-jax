# Copilot Rules

## Code Style

Follow PEP 8 guidelines for Python code.

Use 4 spaces for indentation.

Write clear and concise comments.

## Testing

Use `pytest` for testing.

When I prompt with `test_xxx`, you should generate a test case using the following template

```python
def test_xxx():
    #
    # Givens
    #

    #
    # Whens
    #

    #
    # Thens
    #
    pass
```

For example, if I prompt with `test_alpha`, you should generate a test case called `test_alpha` that follows the template.