"""Quick TUI import check."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Test 1: Import the TUI app
from deepseek_tui.tui import DeepSeekApp
print("✓ DeepSeekApp imported")

# Test 2: Check key bindings
binds = DeepSeekApp.BINDINGS
print(f"✓ {len(binds)} bindings defined")

# Test 3: Import all screens
from deepseek_tui.tui.screens.chat import ChatScreen, DeepSeekInput, ChatMessage
print(f"✓ ChatScreen base: {ChatScreen.__bases__[0].__name__}")

# Test 4: Check DeepSeekInput inherits Input
from textual.widgets import Input
assert issubclass(DeepSeekInput, Input), "DeepSeekInput must inherit Input"
print("✓ DeepSeekInput extends Input")

# Test 5: Check the import chain
from deepseek_tui.tui.screens.sidebar import SidebarPanel
print(f"✓ SidebarPanel: {SidebarPanel.__bases__[0].__name__}")

from deepseek_tui.tui.screens.approval import ApprovalDialog
print(f"✓ ApprovalDialog: {ApprovalDialog.__bases__[0].__name__}")

from deepseek_tui.tui.commands import dispatch, CommandResult, get_registry
from deepseek_tui.tui.commands.all_commands import register_all
register_all()
registry = get_registry()
cmds = registry.all()
print(f"✓ {len(cmds)} commands registered")

print("\nAll TUI imports OK.")
