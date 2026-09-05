import asyncio

from app.documents.job_dispatcher import DocumentJobDispatcher


def test_document_dispatcher_runs_only_one_build_worker_at_a_time() -> None:
    async def exercise() -> int:
        dispatcher = DocumentJobDispatcher()
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        release_first = asyncio.Event()
        completed = asyncio.Event()
        active = 0
        peak_active = 0

        async def first_build() -> None:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            first_started.set()
            await release_first.wait()
            active -= 1

        async def second_build() -> None:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            second_started.set()
            active -= 1
            completed.set()

        await dispatcher.enqueue("first", first_build())
        await first_started.wait()
        await dispatcher.enqueue("second", second_build())
        await asyncio.sleep(0)
        assert not second_started.is_set()

        release_first.set()
        await completed.wait()
        return peak_active

    assert asyncio.run(exercise()) == 1


def test_dispatchers_share_the_single_process_wide_build_worker_slot() -> None:
    async def exercise() -> int:
        legacy_dispatcher = DocumentJobDispatcher()
        candidate_dispatcher = DocumentJobDispatcher()
        legacy_started = asyncio.Event()
        candidate_started = asyncio.Event()
        release_legacy = asyncio.Event()
        completed = asyncio.Event()
        active = 0
        peak_active = 0

        async def legacy_build() -> None:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            legacy_started.set()
            await release_legacy.wait()
            active -= 1

        async def candidate_build() -> None:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            candidate_started.set()
            active -= 1
            completed.set()

        await legacy_dispatcher.enqueue("legacy-build", legacy_build())
        await legacy_started.wait()
        await candidate_dispatcher.enqueue("candidate-build", candidate_build())
        await asyncio.sleep(0)
        assert not candidate_started.is_set()

        release_legacy.set()
        await completed.wait()
        return peak_active

    assert asyncio.run(exercise()) == 1
