import debugpy
from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Markdown, Static, TabbedContent, TabPane, Tabs

TAB_NAMES = (
  "cross",
  "horizontal",
  "custom",
  "left",
  "right",
)


class TabsApp(App):
  """Demonstrates the Tabs widget."""

  CSS_PATH = "tabs.tcss"

  BINDINGS = []

  def compose(self) -> ComposeResult:
    with TabbedContent():
      for tab_name in TAB_NAMES:
        with TabPane(tab_name):
          yield Static(f"Content for {tab_name}", name=tab_name, id=tab_name)
    yield Footer()

  def on_mount(self) -> None:
    """Focus the tabs when the app starts."""
    self.query_one(TabbedContent).focus()


if __name__ == "__main__":
  if __debug__:
    debugpy.listen(("127.0.0.1", 5678))
    debugpy.wait_for_client()
    debugpy.is_client_connected()
  app = TabsApp()
  app.run()
