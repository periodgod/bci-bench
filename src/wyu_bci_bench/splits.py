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


def predefined(mask_train: np.ndarray, mask_test: np.ndarray, name: str,
               mask_valid: np.ndarray | None = None, **info: object) -> Split:
    """把外部已确定的索引/掩码包装成统一 Split，保留调用方提供的协议说明。"""
    return Split(name=name, train=np.asarray(mask_train, bool), test=np.asarray(mask_test, bool),
                 valid=None if mask_valid is None else np.asarray(mask_valid, bool), info=dict(info))


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
