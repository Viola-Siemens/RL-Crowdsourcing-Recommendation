import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as npy
from dateutil.parser import parse


class Data:
    worker_quality: Dict[int, float]
    worker_category: Dict[Tuple[int, int], int]
    worker_project_cnt: Dict[int, int]
    project_info: Dict[int, Dict[str, int]]
    entry_info: Dict[int, Dict[int, Dict[str, int]]]
    industry_list: Dict[str, int]
    n_state: int
    _n_cat: int = 10
    _project_by_time: List[Dict[str, int]]
    _worker_id_rmap: List[int]

    def get_data(self) -> None:
        p_path = os.path.abspath(os.path.dirname(__file__))
        data_dir = p_path[:p_path.rindex('data')] + "resources/"

        # read worker attribute: worker_quality
        worker_quality = {}
        worker_id_rmap = []
        csvfile = open(data_dir + "worker_quality.csv", "r")
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for line in csvreader:
            if float(line[1]) > 0.0:
                worker_id = int(line[0])
                worker_quality[worker_id] = float(line[1]) / 100.0
                worker_id_rmap.append(worker_id)
        csvfile.close()

        # read project id
        file = open(data_dir + "project_list.csv", "r")
        project_list_lines = file.readlines()
        file.close()
        project_dir = data_dir + "project/"
        entry_dir = data_dir + "entry/"

        project_info = {}
        entry_info = {}
        limit = 24
        industry_list = {}
        worker_category = {}
        worker_project_cnt = {}
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

            project_info[project_id] = { "id": project_id }
            project_info[project_id]["sub_category"] = int(text["sub_category"])  # project sub_category 2 ~ 100
            project_info[project_id]["category"] = int(text["category"])  # project category 2 ~ 10
            project_info[project_id]["entry_count"] = int(text["entry_count"])  # project answers in total
            project_info[project_id]["start_date"] = int(parse(text["start_date"]).timestamp())  # project start date
            project_info[project_id]["deadline"] = int(parse(text["deadline"]).timestamp())  # project end date

            if text["industry"] is None:
                text["industry"] = "none"
            if text["industry"] not in industry_list:
                industry_list[text["industry"]] = len(industry_list)
            project_info[project_id]["industry"] = industry_list[text["industry"]]  # project domain

            entry_info[project_id] = {}
            k = 0
            while k < entry_count:
                # print(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r")
                file = open(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r", errors='ignore')
                htmlcode = file.read()
                file.close()
                text = json.loads(htmlcode, strict=False)

                for item in text["results"]:
                    entry_number = int(item["entry_number"])
                    entry_info[project_id][entry_number] = {}
                    entry_info[project_id][entry_number]["entry_created_at"] =\
                        int(parse(item["entry_created_at"]).timestamp())  # worker answer_time
                    worker_id = int(item["entry_number"])
                    entry_info[project_id][entry_number]["worker"] = worker_id
                    tp = (worker_id, project_id)
                    if tp not in worker_category:
                        worker_category[tp] = 0
                    worker_category[tp] += 1
                    if worker_id not in worker_project_cnt:
                        worker_project_cnt[worker_id] = 0
                    worker_project_cnt[worker_id] += 1
                k += limit

        print("finish read_data")
        self.worker_quality = worker_quality
        self.project_info = project_info
        self.entry_info = entry_info
        self.industry_list = industry_list
        self.n_state = self._n_cat + len(self.industry_list)
        pbt: List[Dict[str, int]] = []
        for pid in project_info.keys():
            pbt.append(project_info[pid])
        self._project_by_time = sorted(pbt, key=lambda a: a["start_date"])
        self._worker_id_rmap = worker_id_rmap

    def get_state_array(self, index: int) -> npy.ndarray:
        project = self._project_by_time[index]
        ret = npy.zeros((self.n_state,))
        ret[project["category"] - 1] = 1
        ret[project["industry"] + self._n_cat] = 1
        return ret

    def get_standard_reward(self, worker_id: int, project_id: int) -> float:
        return self.worker_category[(worker_id, project_id)] / self.worker_project_cnt[worker_id]

    def get_quality_reward(self, worker_id: int) -> float:
        return self.worker_quality[worker_id]

    def get_project_id_by_index(self, index: int) -> int:
        return self._project_by_time[index]["id"]

    def get_worker_id_by_index(self, index: int) -> int:
        return self._worker_id_rmap[index]

    def get_projects_length(self) -> int:
        return len(self._project_by_time)
