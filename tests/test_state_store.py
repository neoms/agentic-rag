"""运行时状态持久化：消息/任务落盘、重启恢复、裁剪与清理"""

from datetime import datetime, timedelta, timezone

from src.store.state_store import RuntimeStateStore


def test_messages_persist_across_restart(tmp_path):
    store1 = RuntimeStateStore(tmp_path / "s.db")
    store1.add_message("s1", "user", "q1")
    store1.add_message("s1", "assistant", "a1")

    # 模拟重启：新实例读同一文件
    store2 = RuntimeStateStore(tmp_path / "s.db")
    assert store2.get_messages("s1") == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_trim_session_keeps_recent(tmp_path):
    store = RuntimeStateStore(tmp_path / "s.db")
    for i in range(600):
        store.add_message("s", "user" if i % 2 == 0 else "assistant", f"m{i}")
    store.trim_session("s", 500)
    msgs = store.get_messages("s")
    assert len(msgs) == 500
    assert msgs[0]["content"] == "m100" and msgs[-1]["content"] == "m599"


def test_hallucination_persisted(tmp_path):
    store = RuntimeStateStore(tmp_path / "s.db")
    store.add_message("s", "user", "q")
    store.add_message("s", "assistant", "a")
    store.set_hallucination("s", {"passed": True, "faithfulness": 92.5})
    msgs = store.get_messages("s")
    assert msgs[1]["hallucination"] == {"passed": True, "faithfulness": 92.5}


def test_task_upsert_and_mark_interrupted(tmp_path):
    store = RuntimeStateStore(tmp_path / "s.db")
    now = datetime.now(timezone.utc).isoformat()
    store.upsert_task({
        "task_id": "t1", "doc_id": "d1", "filename": "a.txt",
        "status": "processing", "message": "处理中", "created_at": now,
        "completed_at": None, "chunk_count": 0,
    })
    assert store.get_task("t1")["status"] == "processing"
    assert store.mark_interrupted_tasks() == 1
    assert store.get_task("t1")["status"] == "failed"


def test_prune_tasks_keep_and_ttl(tmp_path):
    store = RuntimeStateStore(tmp_path / "s.db")
    now = datetime.now(timezone.utc).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    for i in range(100):
        store.upsert_task({
            "task_id": f"n{i}", "doc_id": f"d{i}", "filename": f"{i}.txt",
            "status": "completed", "message": "ok", "created_at": now,
            "completed_at": now, "chunk_count": 1,
        })
    # 最旧的 20 个任务（keep 阶段删除）
    for i in range(20):
        tid = f"old-{i}"
        store.upsert_task({
            "task_id": tid, "doc_id": f"sd{i}", "filename": f"s{i}.txt",
            "status": "completed", "message": "ok", "created_at": old,
            "completed_at": old, "chunk_count": 1,
        })
    # 最新但已完成超过 TTL 的 5 个任务（TTL 阶段删除）
    for i in range(5):
        store.upsert_task({
            "task_id": f"ttl-{i}", "doc_id": f"td{i}", "filename": f"t{i}.txt",
            "status": "completed", "message": "ok", "created_at": now,
            "completed_at": old, "chunk_count": 1,
        })
    store.prune_tasks(keep=100, ttl_days=7)
    tasks = store.list_tasks()
    # keep 后剩 100（含 5 条 ttl 占位），TTL 再删 5 条 → 95
    assert len(tasks) == 95
    ids = {t["task_id"] for t in tasks}
    assert not any(x.startswith("old-") for x in ids)   # keep 删除最旧
    assert not any(x.startswith("ttl-") for x in ids)   # TTL 删除已过期
    assert all(f"n{i}" in ids for i in range(5, 100))


def test_clear_messages(tmp_path):
    store = RuntimeStateStore(tmp_path / "s.db")
    store.add_message("s", "user", "q")
    store.clear_messages("s")
    assert store.get_messages("s") == []
