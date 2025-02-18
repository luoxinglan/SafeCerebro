import os
import pickle
import pymysql
import json


class MetricsInserter:
    #TODO：向metrics_TTC插入TTC数据
    def __init__(self, host, user, password, db):
        """
        初始化数据库连接。

        :param host: MySQL 数据库主机地址
        :param user: MySQL 用户名
        :param password: MySQL 密码
        :param db: 要使用的数据库名称
        """
        self.connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db,
            cursorclass=pymysql.cursors.DictCursor  # 使用字典游标以便更容易地处理数据
        )
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        """
        创建 metrics 表（如果尚未存在）。
        """
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            folder_name VARCHAR(255) NOT NULL,
            collision_rate FLOAT,
            avg_red_light_freq FLOAT,
            avg_stop_sign_freq FLOAT,
            out_of_road_length FLOAT,
            route_following_stability FLOAT,
            route_completion FLOAT,
            avg_time_spent FLOAT DEFAULT 0,
            avg_acceleration FLOAT,
            avg_yaw_velocity FLOAT,
            avg_lane_invasion_freq FLOAT,
            safety_os FLOAT,
            task_os FLOAT,
            comfort_os FLOAT,
            final_score FLOAT NOT NULL
        )
        '''
        with self.connection.cursor() as cursor:
            cursor.execute(create_table_query)
        self.connection.commit()

    def insert_metrics_from_folder(self, base_dir):
        """
        遍历指定目录下的每个子文件夹，查找并读取 eval_results/results.pkl 文件中的数据，
        并将数据插入到数据库中。

        :param base_dir: 包含日志文件的根目录路径
        """
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(folder_path):
                file_path = os.path.join(folder_path, 'eval_results', 'results.pkl')
                if os.path.exists(file_path):
                    #TODO：UPDATE12.16：略过已有文件夹
                    if not self._is_folder_processed(folder_name):
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            if isinstance(data, dict):
                                self._insert_metric_data(folder_name, data)

    def _is_folder_processed(self, folder_name):
        """
        检查指定文件夹的数据是否已经存在于数据库中。

        :param folder_name: 文件夹名称
        :return: 如果已存在则返回 True，否则返回 False
        """
        query = "SELECT COUNT(*) FROM metrics WHERE folder_name = %s"
        with self.connection.cursor() as cursor:
            cursor.execute(query, (folder_name,))
            result = cursor.fetchone()
        return result['COUNT(*)'] > 0

    def _insert_metric_data(self, folder_name, data):
        """
        将单个文件的数据插入到数据库中。

        :param folder_name: 文件夹名称
        :param data: 从 .pkl 文件中加载的字典数据
        """
        columns = [
            'folder_name',
            'collision_rate',
            'avg_red_light_freq',
            'avg_stop_sign_freq',
            'out_of_road_length',
            'route_following_stability',
            'route_completion',
            'avg_time_spent',
            'avg_acceleration',
            'avg_yaw_velocity',
            'avg_lane_invasion_freq',
            'safety_os',
            'task_os',
            'comfort_os',
            'final_score'
        ]

        placeholders = ', '.join(['%s'] * len(columns))
        query = f'INSERT INTO metrics ({", ".join(columns)}) VALUES ({placeholders})'
        values = [folder_name] + [data.get(col, None) for col in columns[1:]]

        with self.connection.cursor() as cursor:
            cursor.execute(query, values)
        self.connection.commit()

    def close(self):
        """
        关闭数据库连接。
        """
        self.connection.close()


# 使用示例：
if __name__ == "__main__":
    inserter = MetricsInserter(host='localhost', user='heihuhu', password='1234', db='db')
    inserter.insert_metrics_from_folder('./log/exp')
    inserter.close()



