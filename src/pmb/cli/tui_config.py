"""Interactive TUI for PMB config editing.

Run via `pmb tune`. Shows sidebar of categories, scrollable settings list,
inline editor with type validation. Saves to workspace config.yaml.

Built on textual (terminal UI framework).
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
)

from pmb.config import SCHEMA, Config


def _category_of(key: str) -> str:
    return key.split(".", 1)[0]


def _categories() -> list[str]:
    cats = sorted({_category_of(k) for k in SCHEMA})
    return cats


def _settings_in(category: str) -> list[tuple[str, object]]:
    return sorted(
        ((k, s) for k, s in SCHEMA.items() if _category_of(k) == category),
        key=lambda kv: kv[0],
    )


class EditScreen(ModalScreen):
    """Modal dialog to edit a single setting."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, key: str, current_value, setting):
        super().__init__()
        self.key = key
        self.current_value = current_value
        self.setting = setting

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-dialog"):
            yield Static(f"[bold cyan]Edit:[/] {self.key}", id="edit-title")
            yield Static(f"{self.setting.help}", id="edit-help")
            yield Static(
                f"Type: [yellow]{self.setting.type.__name__}[/]   "
                f"Default: [dim]{self.setting.default!r}[/]   "
                f"Current: [green]{self.current_value!r}[/]",
                id="edit-meta",
            )
            if self.setting.choices:
                yield Static(
                    f"Choices: [magenta]{', '.join(map(str, self.setting.choices))}[/]",
                    id="edit-choices",
                )
            if self.setting.min is not None or self.setting.max is not None:
                yield Static(
                    f"Range: min={self.setting.min!r}, max={self.setting.max!r}",
                    id="edit-range",
                )
            yield Input(
                placeholder=f"new value ({self.setting.type.__name__})",
                value=str(self.current_value),
                id="edit-input",
            )
            yield Static(
                "[dim]Enter to save • Esc to cancel[/]",
                id="edit-hint",
            )

    def on_mount(self) -> None:
        self.query_one("#edit-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PMBTuneApp(App):
    """Main TUI app."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 20;
        background: $boost;
        border-right: solid $primary;
    }

    #sidebar > ListView {
        padding: 1;
    }

    #main {
        width: 1fr;
        padding: 1 2;
    }

    #main-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    DataTable {
        height: 1fr;
    }

    #edit-dialog {
        width: 80;
        height: auto;
        padding: 2 3;
        background: $panel;
        border: thick $accent;
    }

    #edit-title { margin-bottom: 1; }
    #edit-help { color: $text-muted; margin-bottom: 1; }
    #edit-meta { margin-bottom: 1; }
    #edit-choices { margin-bottom: 1; }
    #edit-range { margin-bottom: 1; }
    #edit-input { margin-top: 1; margin-bottom: 1; }
    #edit-hint { color: $text-muted; margin-top: 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reload", "Reload"),
        Binding("enter", "edit_selected", "Edit", show=False),
        Binding("d", "reset_to_default", "Reset to default"),
    ]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.current_category: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("[bold]Categories[/]")
                yield ListView(
                    *(ListItem(Label(cat), id=f"cat-{cat}") for cat in _categories()),
                    id="cat-list",
                )
            with VerticalScroll(id="main"):
                yield Static("PMB tune", id="main-title")
                yield Static(
                    "[dim]Select a category on the left, then press Enter "
                    "on a setting to edit. 'd' resets to default. 'q' quits.[/]",
                    id="main-help",
                )
                yield DataTable(id="settings-table", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#settings-table", DataTable)
        table.add_columns("Key", "Value", "Source", "Type", "Default")
        # Select the first category by default
        cats = _categories()
        if cats:
            self.current_category = cats[0]
            self._populate_table()
            cat_list = self.query_one("#cat-list", ListView)
            cat_list.index = 0

    def _populate_table(self) -> None:
        table = self.query_one("#settings-table", DataTable)
        table.clear()
        if not self.current_category:
            return
        title = self.query_one("#main-title", Static)
        title.update(
            f"[bold cyan]{self.current_category}[/] settings "
            f"([dim]{len(_settings_in(self.current_category))} keys[/])"
        )
        for key, setting in _settings_in(self.current_category):
            value = self.cfg.get(key)
            source = self.cfg.source_of(key)
            source_color = {
                "workspace": "green", "global": "yellow",
                "override": "magenta", "default": "dim",
            }.get(source, "white")
            table.add_row(
                key,
                str(value),
                f"[{source_color}]{source}[/]",
                setting.type.__name__,
                str(setting.default),
                key=key,
            )

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None:
            return
        item_id = event.item.id or ""
        if item_id.startswith("cat-"):
            self.current_category = item_id[len("cat-"):]
            self._populate_table()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        self._edit_row(event.row_key.value)

    def action_edit_selected(self) -> None:
        table = self.query_one("#settings-table", DataTable)
        if table.row_count == 0:
            return
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            self._edit_row(row_key.value)
        except Exception:
            pass

    def _edit_row(self, key: str) -> None:
        if not key or key not in SCHEMA:
            return
        setting = SCHEMA[key]
        current = self.cfg.get(key)

        def _after_edit(new_value):
            if new_value is None or new_value == str(current):
                return
            try:
                self.cfg.set_workspace(key, new_value)
                self._populate_table()
                self.notify(f"saved {key} = {new_value}", severity="information")
            except Exception as e:
                self.notify(f"validation failed: {e}", severity="error")

        self.push_screen(EditScreen(key, current, setting), _after_edit)

    def action_reset_to_default(self) -> None:
        table = self.query_one("#settings-table", DataTable)
        if table.row_count == 0:
            return
        try:
            row_key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key
            key = row_key.value
            if not key or key not in SCHEMA:
                return
            setting = SCHEMA[key]
            self.cfg.set_workspace(key, setting.default)
            self._populate_table()
            self.notify(f"reset {key} → {setting.default!r}", severity="information")
        except Exception as e:
            self.notify(f"reset failed: {e}", severity="error")

    def action_reload(self) -> None:
        # Reload config from disk
        from pmb.config import Config as _C
        self.cfg = _C(
            workspace_dir=self.cfg.workspace_dir,
            pmb_home=self.cfg.pmb_home,
        )
        self._populate_table()
        self.notify("reloaded", severity="information")

    def action_quit(self) -> None:
        self.exit()


def run_tui(cfg: Config) -> None:
    """Entry point: launch the TUI."""
    PMBTuneApp(cfg).run()
