"""Verify Input.Submitted works correctly."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from textual.widgets import Input

# Test direct construction
msg = Input.Submitted("hello")
print(f"✓ Input.Submitted created: value={msg.value}")

# Test construction via self
class TestInput(Input):
    def test(self):
        m = Input.Submitted("test-value")
        print(f"✓ Input.Submitted from method: value={m.value}")
        assert m.value == "test-value", f"Expected 'test-value', got '{m.value}'"

t = TestInput()
print("✓ TestInput created")
# Can't call test() without mounting, but direct construction works

print("\nAll checks passed.")
