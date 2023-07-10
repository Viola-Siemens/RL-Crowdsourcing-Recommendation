from typing import List, Tuple

import numpy as npy
import json
import os
import csv
from dateutil.parser import parse
from data.Action import Action


class Environment:
    _n_project: int
    _n_worker: int
    _active_project: npy.ndarray
    _active_worker: npy.ndarray
    _init_active_project: npy.ndarray
    _init_active_worker: npy.ndarray
    _buffered_states: List[npy.ndarray]
    _done: bool
    # TODO: 将读入的数据类放在成员这里

    def __init__(self, n_project: int, n_worker: int, active_project: List[int], active_worker: List[int]):
        self._n_project = n_project
        self._n_worker = n_worker
        self._init_active_project = npy.zeros(shape=(n_project,))
        self._init_active_worker = npy.zeros(shape=(n_worker,))
        for p in active_project:
            self._init_active_project[p] = 1
        for w in active_worker:
            self._init_active_worker[w] = 1
        self._buffered_states = []
        self._done = False

    def reset(self) -> None:
        self._active_project = self._init_active_project.copy()
        self._active_worker = self._init_active_worker.copy()
        pass

    def sample(self) -> Action:
        # TODO 随机选择一个可行的行为
        pass

    def perform(self, action: Action) -> float:
        # TODO 执行行为获得奖励值
        pass

    def get_state(self) -> Tuple[npy.ndarray, npy.ndarray]:
        return self._active_project, self._active_worker

    def get_history_states(self, n: int) -> List[npy.ndarray]:
        if len(self._buffered_states) < n:
            zero_paddings = [npy.zeros(shape=(self._n_project + self._n_worker,))] * (n - len(self._buffered_states))
            return zero_paddings + self._buffered_states
        return self._buffered_states[-n:]

    def get_data(self) -> Tuple[dict, dict, dict, dict]:
        p_path = os.path.abspath(os.path.dirname(__file__))
        data_dir = p_path[:p_path.rindex('data')]+"dataset/"

        ## read worker attribute: worker_quality
        worker_quality = {}
        csvfile = open(data_dir + "worker_quality.csv", "r")
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for line in csvreader:
            if float(line[1]) > 0.0:
                worker_quality[int(line[0])] = float(line[1]) / 100.0
        csvfile.close()

        ## read project id
        file = open(data_dir + "project_list.csv", "r")
        project_list_lines = file.readlines()
        file.close()
        project_dir = data_dir + "project/"
        entry_dir = data_dir + "entry/"

        all_begin_time = parse("2018-01-01T0:0:0Z")

        project_info = {}
        entry_info = {}
        limit = 24
        industry_list = {}
        # print(project_list_lines)

        for line in project_list_lines:
            line = line.strip('\n').split(',')

            project_id = int(line[0])
            entry_count = int(line[1])
            # print(project_dir + "project_" + str(project_id) + ".txt")
            file = open(project_dir + "project_" + str(project_id) + ".txt", "r", errors='ignore')
            htmlcode = file.read()
            file.close()
            text = json.loads(htmlcode)

            project_info[project_id] = {}
            project_info[project_id]["sub_category"] = int(text["sub_category"])  # project sub_category
            project_info[project_id]["category"] = int(text["category"])  # project category
            project_info[project_id]["entry_count"] = int(text["entry_count"])  # project answers in total
            project_info[project_id]["start_date"] = parse(text["start_date"])  # project start date
            project_info[project_id]["deadline"] = parse(text["deadline"])  # project end date

            if text["industry"] not in industry_list:
                industry_list[text["industry"]] = len(industry_list)
            project_info[project_id]["industry"] = industry_list[text["industry"]]  # project domain

            entry_info[project_id] = {}
            k = 0
            while (k < entry_count):
                # print(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r")
                file = open(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r", errors='ignore')
                htmlcode = file.read()
                file.close()
                text = json.loads(htmlcode, strict=False)

                for item in text["results"]:
                    entry_number = int(item["entry_number"])
                    entry_info[project_id][entry_number] = {}
                    entry_info[project_id][entry_number]["entry_created_at"] = parse(
                        item["entry_created_at"])  # worker answer_time
                    entry_info[project_id][entry_number]["worker"] = int(item["entry_number"])  # work_id
                k += limit

        print("finish read_data")
        return worker_quality, project_info, entry_info, industry_list

if __name__ == "__main__":
    environment = Environment(4, 3, [0,1,0,1], [1,0,0])
    environment.get_data()