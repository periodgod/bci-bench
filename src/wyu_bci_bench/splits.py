"""Group-aware split protocols.

Every split is returned as boolean masks over the dataset rows so it can be
saved, audited and re-applied exactly. Window-level shuffling across people is
never offered here: it leaks participant identity into the test set.
"""

# 为什么全库都返回行掩码：掩码能直接应用到 X、y 和元数据，也能写入报告供人
# 复核。切分函数只定义“哪些行属于哪一侧”，不负责训练模型或选择超参数。

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    """一次数据划分。

    ``train``、``test``、``valid`` 均为与原数据行数相等的布尔数组。验证集
    ``valid`` 与测试集 ``test`` 的职责不同：验证集用于训练期选模型/超参数，
    测试集只用于最终报告。``info`` 保存人可读的协议元数据。
    """
    name: str
    train: np.ndarray
    test: np.ndarray
    valid: np.ndarray | None = None
    info: dict[str, object] = field(default_factory=dict)

    def check_disjoint(self, groups: np.ndarray | None = None) -> None:
        """检查各集合不共享样本；提供 ``groups`` 时也检查不共享组。

        例如 LOSO 中应将参与者编号作为 groups，这样即使切分掩码无行重叠，
        也能发现同一参与者是否同时进入训练和测试。
        """
        parts = [("train", self.train), ("test", self.test)]
        if self.valid is not None:
            parts.append(("valid", self.valid))
        for i, (a_name, a) in enumerate(parts):
            for b_name, b in parts[i + 1:]:
                if np.any(a & b):
                    raise AssertionError(f"{self.name}: {a_name} and {b_name} share rows")
                if groups is not None and set(groups[a].tolist()) & set(groups[b].tolist()):
                    raise AssertionError(f"{self.name}: {a_name} and {b_name} share groups")


def loso(subjects: np.ndarray, people: Sequence[str] | None = None) -> Iterator[Split]:
    """逐个留出参与者的外层交叉验证（Leave-One-Subject-Out）。

    每轮一个参与者的所有窗口进入测试，其余参与者进入训练。它评估跨参与者
    泛化；不能将随机打散窗口的切分结果称作 LOSO。``people`` 可指定参与者子集。
    """
    subjects = np.asarray(subjects)
    for held in people if people is not None else sorted(set(subjects.tolist())):
        test = subjects == held
        yield Split(name=f"loso:{held}", train=~test, test=test, info={"held_out": held})


def inner_loso(subjects: np.ndarray, outer_train: np.ndarray) -> Iterator[Split]:
    """只在外层训练参与者中再做 LOSO，用于内层模型选择。

    这样外层留出者的信息不会参与 C、训练轮数等超参数选择，防止测试数据泄漏。
    返回的 ``test`` 是当前内层验证参与者，因此这里虽沿用 Split 的 test 字段，
    实际角色是验证折。
    """
    subjects = np.asarray(subjects)
    for held in sorted(set(subjects[outer_train].tolist())):
        valid = outer_train & (subjects == held)
        yield Split(name=f"inner:{held}", train=outer_train & ~valid, test=valid, info={"held_out": held})


def forward_blocks(keys, eligible, recording_order, fractions=((.5, .75), (.75, 1.))):
    """Expanding-prefix inner folds restricted to eligible outer training rows.

    Each pair is (train_end, validation_end) as a fraction of eligible rows in
    each stream. The explicit numeric order must be unique within a stream.
    """
    keys, eligible, order = np.asarray(keys), np.asarray(eligible, bool), np.asarray(recording_order)
    if keys.ndim != 1 or eligible.shape != keys.shape or order.shape != keys.shape or not np.isfinite(order).all():
        raise ValueError("keys, eligibility and finite numeric order must align")
    for left, right in fractions:
        if not 0 < left < right <= 1:
            raise ValueError("expected 0 < train_end < validation_end <= 1")
        train, valid = np.zeros(len(keys), bool), np.zeros(len(keys), bool)
        for key in np.unique(keys[eligible]):
            ix = np.flatnonzero(eligible & (keys == key))
            if len(np.unique(order[ix])) != len(ix):
                raise ValueError("recording order has ties within a stream")
            ix = ix[np.argsort(order[ix], kind="stable")]
            a, b = int(len(ix) * left), int(len(ix) * right)
            if not 0 < a < b:
                raise ValueError("not enough eligible rows for forward blocks")
            train[ix[:a]], valid[ix[a:b]] = True, True
        split = Split(f"forward:{left}-{right}", train, valid,
                      info={"train_end": left, "validation_end": right})
        split.check_disjoint()
        yield split


def cross_session(
    subjects: np.ndarray,
    sessions: np.ndarray,
    subject: str,
    train_sessions: Sequence[str] = ("ses-1",),
    test_sessions: Sequence[str] | None = None,
) -> Split:
    """同一参与者内以早期会话校准、在后续会话评估。

    默认测试所有非训练会话；显式传入 ``test_sessions`` 时只测试列出的会话。
    该协议衡量跨会话迁移，不等价于跨参与者 LOSO。
    """
    subjects, sessions = np.asarray(subjects), np.asarray(sessions)
    own = subjects == subject
    train = own & np.isin(sessions, list(train_sessions))
    test = own & ~train if test_sessions is None else own & np.isin(sessions, list(test_sessions))
    return Split(name=f"xsession:{subject}", train=train, test=test,
                 info={"subject": subject, "train_sessions": list(train_sessions),
                       "test_sessions": sorted(set(sessions[test].tolist()))})


def ordered_kshot_mask(subjects: np.ndarray, sessions: np.ndarray, y: np.ndarray,
                       subject: str, k: int, *, train_sessions: Sequence[str] = ("ses-1",),
                       n_classes: int | None = None,
                       recording_order: np.ndarray | None = None) -> np.ndarray:
    """Select the first k examples per class by explicit recording order.

    If ``recording_order`` is omitted, input row order is treated as recording
    order. Ties retain the original row order.
    """
    if k < 1:
        raise ValueError("k must be positive")
    subjects, sessions, y = np.asarray(subjects), np.asarray(sessions), np.asarray(y)
    if not (len(subjects) == len(sessions) == len(y)):
        raise ValueError("subjects, sessions and y must have equal length")
    order = np.arange(len(y)) if recording_order is None else np.asarray(recording_order)
    if len(order) != len(y) or order.ndim != 1 or not np.issubdtype(order.dtype, np.number):
        raise ValueError("recording_order must be a numeric vector aligned with y")
    candidate = (subjects == subject) & np.isin(sessions, list(train_sessions))
    classes = range(int(y.max()) + 1) if n_classes is None else range(int(n_classes))
    out = np.zeros(len(y), dtype=bool)
    for cls in classes:
        rows = np.flatnonzero(candidate & (y == cls))
        rows = rows[np.argsort(order[rows], kind="stable")]
        out[rows[:k]] = True
    return out


def predefined(mask_train: np.ndarray, mask_test: np.ndarray, name: str,
               mask_valid: np.ndarray | None = None, **info: object) -> Split:
    """把外部已确定的索引/掩码包装成统一 Split，保留调用方提供的协议说明。"""
    return Split(name=name, train=np.asarray(mask_train, bool), test=np.asarray(mask_test, bool),
                 valid=None if mask_valid is None else np.asarray(mask_valid, bool), info=dict(info))


def chronological_cell_holdout(*cell_groups: np.ndarray, test_fraction: float = 1 / 3) -> Split:
    """每个分组单元（如 participant×session×interface）内按行顺序留出末尾窗口。

    行顺序必须是记录顺序。每个单元的后 ``test_fraction`` 进入测试，其余进入训练；
    所有人一起拟合。它模拟“已见参与者的后期数据”，但仍是会话内留出，比跨会话乐观。
    """
    if not cell_groups:
        raise ValueError("need at least one grouping column")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    keys = np.asarray(["|".join(map(str, row)) for row in zip(*(np.asarray(g) for g in cell_groups))])
    test = np.zeros(len(keys), dtype=bool)
    for key in dict.fromkeys(keys.tolist()):
        ix = np.flatnonzero(keys == key)
        n_test = int(round(len(ix) * test_fraction))
        if n_test:
            test[ix[len(ix) - n_test:]] = True
    return Split(name="chronological_cell_holdout", train=~test, test=test,
                 info={"test_fraction": float(test_fraction), "cells": int(len(set(keys.tolist())))})


def seen_later_session(subjects: np.ndarray, sessions: np.ndarray, test_session: str,
                       people: Sequence[str]) -> Split:
    """已见参与者跨会话：``people`` 的 ``test_session`` 进入测试，其余全部（含本人早期会话）训练。"""
    subjects, sessions = np.asarray(subjects), np.asarray(sessions)
    test = np.isin(subjects, list(people)) & (sessions == test_session)
    return Split(name=f"seen_later:{test_session}", train=~test, test=test,
                 info={"people": list(people), "test_session": test_session})


def neuralbench_sealed(subjects: np.ndarray, sessions: np.ndarray, evaluation_subjects: Sequence[str],
                       sealed_sessions: Sequence[str] = ("ses-4", "ses-5", "ses-6")) -> Split:
    """构造 Track 2 的留出会话掩码。

    只有 ``evaluation_subjects`` 在 ``sealed_sessions`` 中的行进入测试，其余数据
    都标为训练。真实提交前必须把参与者名单与赛题合同核对；这个函数不会检查
    上游是否误把密封数据用于特征拟合。
    """
    subjects, sessions = np.asarray(subjects), np.asarray(sessions)
    test = np.isin(subjects, list(evaluation_subjects)) & np.isin(sessions, list(sealed_sessions))
    return Split(name="neuralbench_sealed", train=~test, test=test,
                 info={"evaluation_subjects": list(evaluation_subjects), "sealed_sessions": list(sealed_sessions)})


def subject_ratio(subjects: np.ndarray, seed: int, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Split:
    """Subject-isolated train/valid/test split (OmniEEG uses 8:1:1 with three organiser seeds).

    划分单位是参与者而不是窗口，所以同一人的所有样本只会出现在一个集合。
    这是本地近似切分；组织方公布正式索引后必须使用正式索引，不能把本函数
    生成的切分冒充官方划分。随机种子保存在 ``info``，便于复现。
    """
    subjects = np.asarray(subjects)
    people = np.array(sorted(set(subjects.tolist())))
    order = np.random.default_rng(seed).permutation(len(people))
    n_test = max(1, int(round(ratios[2] * len(people))))
    n_valid = max(1, int(round(ratios[1] * len(people))))
    test_p = set(people[order[:n_test]].tolist())
    valid_p = set(people[order[n_test:n_test + n_valid]].tolist())
    test = np.isin(subjects, list(test_p))
    valid = np.isin(subjects, list(valid_p))
    split = Split(name=f"subject_ratio:seed{seed}", train=~(test | valid), test=test, valid=valid,
                  info={"seed": seed, "ratios": list(ratios), "test_subjects": sorted(test_p),
                        "valid_subjects": sorted(valid_p)})
    split.check_disjoint(subjects)
    return split


def multi_seed_subject_ratio(subjects: np.ndarray, seeds: Sequence[int],
                             ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> list[Split]:
    """对多个种子重复参与者级比例切分，返回逐种子的独立 Split。"""
    return [subject_ratio(subjects, s, ratios) for s in seeds]


def subject_kfold(subjects: np.ndarray, n_splits: int = 5, *, seed: int = 20261114,
                  valid_fraction: float = .2) -> list[Split]:
    """Every participant is tested once; validation comes only from outer training.

    Shuffle the sorted participant IDs once, partition them into near-equal test
    folds, then reserve a seeded participant subset inside each remaining fold
    for validation. No labels or sample counts influence assignment. This is
    outer K-fold with a single inner holdout, not exhaustive nested K-fold.
    """
    subjects = np.asarray(subjects)
    if subjects.ndim != 1 or not len(subjects):
        raise ValueError('subjects must be a nonempty vector')
    people = np.array(sorted(set(subjects.tolist())))
    if not isinstance(n_splits, int) or not 2 <= n_splits <= len(people):
        raise ValueError('n_splits must be between2 and number of participants')
    if not 0 < valid_fraction < 1:
        raise ValueError('valid_fraction must be in(0,1)')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(people)
    folds = []
    for fold, test_people in enumerate(np.array_split(shuffled, n_splits)):
        outer_train = people[~np.isin(people, test_people)]
        if len(outer_train) < 2:
            raise ValueError('each outer training fold needs at least2 participants for train/valid')
        n_valid = min(len(outer_train)-1, max(1, int(round(len(outer_train)*valid_fraction))))
        valid_people = rng.permutation(outer_train)[:n_valid]
        test, valid = np.isin(subjects, test_people), np.isin(subjects, valid_people)
        split = Split(name=f'subject_kfold:{fold+1}/{n_splits}:seed{seed}',
            train=~(test|valid), test=test, valid=valid,
            info={'fold': fold+1, 'n_splits': n_splits, 'seed': seed,
                  'valid_fraction_of_outer_training': valid_fraction,
                  'test_subjects': sorted(test_people.tolist()),
                  'valid_subjects': sorted(valid_people.tolist()),
                  'train_subjects': sorted(set(subjects[~(test|valid)].tolist()))})
        split.check_disjoint(subjects)
        folds.append(split)
    assert np.all(np.stack([split.test for split in folds]).sum(0) == 1)
    return folds
