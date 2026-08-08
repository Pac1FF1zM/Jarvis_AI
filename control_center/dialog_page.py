"""Minimal user-facing dialogue history and personal-memory controls."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from memory.conversations import ConversationStore
from memory.long_term import LongTermMemory


def _label(text: str, object_name: str | None = None) -> QLabel:
    label = QLabel(text)
    if object_name:
        label.setObjectName(object_name)
    return label


class DialogHistoryPage(QWidget):
    def __init__(self, db_path: str | Path, profile_id: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.db_path = str(Path(db_path).resolve())
        self.profile_id = profile_id
        self.store = ConversationStore(self.db_path, profile_id=profile_id)
        self._selected_chat = ""
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(2500)
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 28)
        layout.setSpacing(12)
        layout.addWidget(_label("DIALOGUE MEMORY // 07", "eyebrow"))
        header = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(2)
        title.addWidget(_label("Диалоги", "title"))
        title.addWidget(_label("История доступна для чтения, но старые чаты не подмешиваются в новый сеанс.", "muted"))
        header.addLayout(title)
        header.addStretch(1)
        refresh = QPushButton("Обновить")
        refresh.clicked.connect(self.refresh)
        clear_memory = QPushButton("ОЧИСТИТЬ ПАМЯТЬ")
        clear_memory.setObjectName("danger")
        clear_memory.clicked.connect(self._clear_memory)
        clear_history = QPushButton("Очистить историю")
        clear_history.clicked.connect(self._clear_history)
        header.addWidget(refresh)
        header.addWidget(clear_history)
        header.addWidget(clear_memory)
        layout.addLayout(header)

        self.memory_summary = _label("Профиль памяти загружается…", "memorySummary")
        self.memory_summary.setWordWrap(True)
        layout.addWidget(self.memory_summary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        list_card = QFrame()
        list_card.setObjectName("chatListCard")
        list_layout = QVBoxLayout(list_card)
        list_layout.setContentsMargins(12, 12, 12, 12)
        list_layout.addWidget(_label("ЧАТЫ ПО ДАТЕ", "eyebrow"))
        self.chat_list = QListWidget()
        self.chat_list.setObjectName("chatList")
        self.chat_list.currentItemChanged.connect(self._chat_selected)
        list_layout.addWidget(self.chat_list, 1)
        splitter.addWidget(list_card)

        message_card = QFrame()
        message_card.setObjectName("chatMessageCard")
        message_layout = QVBoxLayout(message_card)
        message_layout.setContentsMargins(16, 14, 16, 14)
        self.chat_title = _label("Выберите диалог", "sectionTitle")
        self.messages = QTextBrowser()
        self.messages.setObjectName("chatMessages")
        self.messages.setOpenExternalLinks(False)
        message_layout.addWidget(self.chat_title)
        message_layout.addWidget(self.messages, 1)
        splitter.addWidget(message_card)
        splitter.setSizes([330, 850])
        layout.addWidget(splitter, 1)

    def refresh(self) -> None:
        selected = self._selected_chat
        chats = self.store.list_chats()
        self.chat_list.blockSignals(True)
        self.chat_list.clear()
        selected_row = -1
        for index, chat in enumerate(chats):
            try:
                stamp = datetime.fromisoformat(chat.updated_at).astimezone().strftime("%d.%m.%Y  %H:%M")
            except ValueError:
                stamp = chat.updated_at
            item = QListWidgetItem(f"{chat.title}\n{stamp}")
            item.setData(Qt.ItemDataRole.UserRole, chat.id)
            self.chat_list.addItem(item)
            if chat.id == selected:
                selected_row = index
        self.chat_list.blockSignals(False)
        if selected_row >= 0:
            self.chat_list.setCurrentRow(selected_row)
        elif chats:
            self.chat_list.setCurrentRow(0)
        else:
            self._selected_chat = ""
            self.chat_title.setText("Диалогов пока нет")
            self.messages.setHtml("<p style='color:#8493a6'>Запустите Jarvis и начните разговор.</p>")
        self._refresh_memory_summary()

    def _chat_selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        chat_id = str(current.data(Qt.ItemDataRole.UserRole) or "")
        self._selected_chat = chat_id
        self.chat_title.setText(current.text().splitlines()[0])
        blocks: list[str] = []
        for message in self.store.messages(chat_id):
            is_user = message.role == "user"
            name = "ВЫ" if is_user else "JARVIS"
            color = "#f7e928" if is_user else "#19e6f2"
            blocks.append(
                "<div style='margin:10px 2px;padding:10px 12px;background:#0b111b;'>"
                f"<div style='color:{color};font-size:10px;font-weight:700'>{name}</div>"
                f"<div style='color:#edf4f8;margin-top:5px'>{html.escape(message.content)}</div>"
                "</div>"
            )
        self.messages.setHtml("".join(blocks))
        self.messages.verticalScrollBar().setValue(self.messages.verticalScrollBar().maximum())

    def _refresh_memory_summary(self) -> None:
        memory = LongTermMemory(self.db_path, profile_id=self.profile_id)
        try:
            facts = [fact.object for fact in memory.personal_facts()]
            apps = [name for name, _count in memory.popular_applications()]
        finally:
            memory.close()
        fact_text = " · ".join(facts) if facts else "важных фактов пока нет"
        apps_text = ", ".join(apps) if apps else "нет"
        self.memory_summary.setText(f"ДОЛГОСРОЧНАЯ ПАМЯТЬ  //  {fact_text}\nТОП ПРИЛОЖЕНИЙ  //  {apps_text}")

    def _clear_memory(self) -> None:
        if QMessageBox.question(
            self,
            "Очистить память",
            "Удалить важные факты и список популярных приложений текущего пользователя? История диалогов останется.",
        ) != QMessageBox.StandardButton.Yes:
            return
        memory = LongTermMemory(self.db_path, profile_id=self.profile_id)
        try:
            memory.clear_profile()
        finally:
            memory.close()
        self._refresh_memory_summary()

    def _clear_history(self) -> None:
        if QMessageBox.question(self, "Очистить историю", "Удалить все сохранённые чаты текущего пользователя?") != QMessageBox.StandardButton.Yes:
            return
        self.store.clear_history()
        self.refresh()

    def shutdown(self) -> None:
        self._timer.stop()
        self.store.close()
