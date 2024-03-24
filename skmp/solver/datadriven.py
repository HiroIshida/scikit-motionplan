import json
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import tqdm
from sklearn.neighbors import BallTree

from skmp.solver.interface import (
    AbstractScratchSolver,
    AbstractSolver,
    ConfigProtocol,
    Problem,
    ResultProtocol,
)
from skmp.trajectory import Trajectory

ConfigT = TypeVar("ConfigT", bound="ConfigProtocol")
ResultT = TypeVar("ResultT", bound="ResultProtocol")

DataT = TypeVar("DataT")


@dataclass
class ChunkedLibrary(Generic[DataT]):
    chunk_list: List[Optional[List[DataT]]]  # at least last chunk is not None
    edit_flags: List[bool]
    file_path: Path
    n_per_chunk: int
    _td_path: Path

    def __post_init__(self):
        assert self.file_path.suffix == ".tar"
        assert self.n_per_chunk > 0
        assert self._td_path.is_dir()

    def __del__(self):
        shutil.rmtree(self._td_path)

    @classmethod
    def create(cls, file_path: Union[Path, str], n_per_chunk: int = 32768) -> "ChunkedLibrary":
        if isinstance(file_path, str):
            file_path = Path(file_path)
        td = tempfile.mkdtemp()
        td_path = Path(td)
        return cls([[]], [True], file_path, n_per_chunk, td_path)

    @classmethod
    def load(cls, file_path: Union[Path, str]) -> "ChunkedLibrary":
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert file_path.suffix == ".tar"
        td = tempfile.mkdtemp()
        td_path = Path(td)
        subprocess.run(["tar", "-xf", str(file_path), "-C", str(td_path)], check=True)

        metadata_path = td_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        n_per_chunk = metadata["n_per_chunk"]
        n_chunk = metadata["n_chunks"]
        chunk_list: List[Optional[List[DataT]]]
        if n_chunk == 0:
            chunk_list = []
        else:
            chunk_list = [None] * (n_chunk - 1)
            last_chunk_path = td_path / f"chunk_{n_chunk - 1}.pkl"
            with open(last_chunk_path, "rb") as f:
                chunk = pickle.load(f)
                assert (
                    len(chunk) == metadata["n_last_chunk"]
                ), f"{len(chunk)} != {metadata['n_last_chunk']}"
                chunk_list.append(chunk)
        edit_flags = [False] * n_chunk
        return cls(chunk_list, edit_flags, file_path, n_per_chunk, td_path)

    def __getitem__(self, idx: int) -> DataT:
        if idx < 0:
            n_data = len(self)
            idx += n_data
        chunk_idx, idx_in_chunk = divmod(idx, self.n_per_chunk)
        if self.chunk_list[chunk_idx] is None:
            file_path = self._td_path / f"chunk_{chunk_idx}.pkl"
            with open(file_path, "rb") as f:
                self.chunk_list[chunk_idx] = pickle.load(f)
        chunk_this = self.chunk_list[chunk_idx]
        assert chunk_this is not None
        return chunk_this[idx_in_chunk]

    def __len__(self) -> int:
        if len(self.chunk_list) == 0:
            return 0
        last_chunk = self.chunk_list[-1]
        assert last_chunk is not None
        return (len(self.chunk_list) - 1) * self.n_per_chunk + len(last_chunk)

    def append(self, data: DataT) -> None:
        n = len(self)
        idx_chunk, idx_in_chunk = divmod(n, self.n_per_chunk)

        need_more_chunk = idx_chunk == len(self.chunk_list)
        if need_more_chunk:
            self.chunk_list.append([])
            self.edit_flags.append(True)

        self.edit_flags[idx_chunk] = True
        self.chunk_list[idx_chunk].append(data)  # type: ignore[union-attr]

    def save(self) -> None:
        n = len(self)
        idx_chunk, idx_in_chunk = divmod(n, self.n_per_chunk)
        if idx_in_chunk == 0:
            n_chunk = idx_chunk
            n_last_chunk = self.n_per_chunk
        else:
            n_chunk = idx_chunk + 1
            n_last_chunk = idx_in_chunk

        metadata_path = self._td_path / "metadata.json"
        with open(metadata_path, "w") as f:
            idx_chunk, idx_in_chunk = divmod(len(self), self.n_per_chunk)
            json_data = {
                "n_chunks": n_chunk,
                "n_last_chunk": n_last_chunk,
                "n_per_chunk": self.n_per_chunk,
            }
            json.dump(json_data, f)

        for i in tqdm.tqdm(range(n_chunk), desc="saving"):
            if not self.edit_flags[i]:
                continue
            file_path = self._td_path / f"chunk_{i}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(self.chunk_list[i], f)

        subprocess.run(
            ["tar", "-cf", str(self.file_path), "-C", str(self._td_path), "."], check=True
        )


@dataclass
class NearestNeigborSolver(AbstractSolver[ConfigT, ResultT, np.ndarray]):
    config: ConfigT
    internal_solver: AbstractScratchSolver[ConfigT, ResultT]
    tree: BallTree
    trajectories: List[Optional[Trajectory]]  # None means no trajectory is available
    knn: int
    infeasibility_threshold: int
    conservative: bool
    previous_est_positive: Optional[bool] = None
    previous_false_positive: Optional[bool] = None

    @classmethod
    def from_chunked_library(
        cls: Type["NearestNeigborSolver[ConfigT, ResultT]"],
        cfdataset: ChunkedLibrary[Tuple[float, np.ndarray, Optional[Trajectory]]],
        n_data_use: int,
        solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,
        knn: int = 1,
        leaf_size: int = 5,
        infeasibility_threshold: Optional[int] = None,
        conservative: bool = False,
    ) -> "NearestNeigborSolver[ConfigT, ResultT]":
        dataset = [cfdataset[i][1:] for i in range(n_data_use)]
        return cls.init(
            solver_type, config, dataset, knn, leaf_size, infeasibility_threshold, conservative
        )

    @classmethod
    def init(
        cls: Type["NearestNeigborSolver[ConfigT, ResultT]"],
        solver_type: Type[AbstractScratchSolver[ConfigT, ResultT]],
        config: ConfigT,  # for internal solver
        dataset: List[Tuple[np.ndarray, Optional[Trajectory]]],
        knn: int = 1,
        leaf_size: int = 5,
        infeasibility_threshold: Optional[int] = None,
        conservative: bool = False,
    ) -> "NearestNeigborSolver[ConfigT, ResultT]":
        assert knn > 0

        tmp, trajectories = zip(*dataset)
        vec_descs = np.array(tmp)
        internal_solver = solver_type.init(config)
        tree = BallTree(vec_descs, leaf_size=leaf_size)

        if infeasibility_threshold is None:
            # do leave-one-out cross validation
            # see the V.A of the following paper for the detail
            # Hauser, Kris. "Learning the problem-optimum map: Analysis and application to global optimization in robotics." IEEE Transactions on Robotics 33.1 (2016): 141-152.
            # actually, the threshold can be tuned wrt specified fp-rate, but what we vary is only integer
            # we have little control over the fp-rate. So just use the best threshold in terms of the accuracy
            errors = []
            thresholds = list(range(1, knn + 1))
            for threshold in thresholds:
                error = 0
                for i, (desc, traj) in enumerate(dataset):
                    k_nearests = tree.query(np.array([desc]), k=knn + 1, return_distance=False)[0][
                        1:
                    ]
                    none_count_in_knn = sum(1 for idx in k_nearests if trajectories[idx] is None)
                    seems_infeasible = none_count_in_knn >= threshold
                    actually_infeasible = traj is None
                    if seems_infeasible != actually_infeasible:
                        error += 1
                errors.append(error)
            print(f"t-error pairs: {list(zip(thresholds, errors))}")
            infeasibility_threshold = thresholds[np.argmin(errors)]
        return cls(
            config,
            internal_solver,
            tree,
            list(trajectories),
            knn,
            infeasibility_threshold,
            conservative,
        )

    def _knn_trajectories(self, query_desc: np.ndarray) -> List[Optional[Trajectory]]:
        # sqdists = np.sum((self.vec_descs - query_desc) ** 2, axis=1)
        # k_nearests = np.argsort(sqdists)[: self.knn]
        k_nearests = self.tree.query(np.array([query_desc]), k=self.knn, return_distance=False)[0]
        return [self.trajectories[i] for i in k_nearests]

    def _solve(self, query_desc: Optional[np.ndarray] = None) -> ResultT:
        if query_desc is not None:
            trajs = self._knn_trajectories(query_desc)
            trajs_without_none = [traj for traj in trajs if traj is not None]
            count_none = len(trajs) - len(trajs_without_none)
            seems_infeasible = count_none >= self.infeasibility_threshold

            if self.conservative:
                self.previous_est_positive = not seems_infeasible

            if self.conservative and seems_infeasible:
                return self.get_result_type().abnormal()

            for guiding_traj in trajs_without_none:
                if guiding_traj is not None:
                    result = self.internal_solver._solve(guiding_traj)
                    if result.traj is not None:

                        if self.conservative:
                            self.previous_false_positive = False

                        return result
            if self.conservative:
                self.previous_false_positive = True

            return self.get_result_type().abnormal()
        else:
            reuse_traj = None
            result = self.internal_solver._solve(reuse_traj)
            return result

    def get_result_type(self) -> Type[ResultT]:
        return self.internal_solver.get_result_type()

    def _setup(self, problem: Problem) -> None:
        self.internal_solver.setup(problem)
        self.previous_est_positive = None
        self.previous_false_positive = None
