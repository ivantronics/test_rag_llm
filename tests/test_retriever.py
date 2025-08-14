import json
from pathlib import Path
from app.retriever import Retriever

def test_retriever_top1_matches_expected():
    kb_dir = str(Path(__file__).resolve().parents[1] / "dataset" / "kb")
    r = Retriever(kb_dir)
    # Каноничные запросы по всем кейсам и ожидаемые документы для них
    cases = [
        ("Сотрудник забыл доменный пароль. Что делать по шагам?", "password_reset_policy.md"),
        ("Какой SSID для персонала в магазине и чем шифруется?", "store_wifi_setup.md"),
        ("Принтер чеков печатает иероглифы. Как исправить?", "pos_printer_troubleshooting.md"),
        ("У сотрудника часто обрывается VPN. Что проверить в первую очередь?", "vpn_access_retail.md"),
        ("Как правильно клеить стикеры с Asset Tag и что писать в CMDB?", "asset_tagging_guidelines.md"),
    ]
    for q, expected in cases:
        top = r.retrieve(q, top_k=1)
        assert top and top[0][0] == expected, f"Expected {expected}, got {top[:1]}"