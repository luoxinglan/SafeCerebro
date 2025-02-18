import os
import pickle
import json
import pymysql
import math


class MetricsTTCInserter:
    def __init__(self, host, user, password, db):
        """
        初始化 MetricsTTCInserter 类，传入数据库连接参数。

        :param host: 数据库主机地址。
        :param user: 数据库用户名。
        :param password: 数据库密码。
        :param db: 数据库名称。
        """
        self.host = host
        self.user = user
        self.password = password
        self.db = db
        self.connection = None

    def connect(self):
        """
        建立与 MySQL 数据库的连接。
        """
        try:
            self.connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.db,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print("数据库连接已建立。")
        except pymysql.MySQLError as e:
            print(f"连接数据库时出错: {e}")
            raise

    def close(self):
        """
        关闭数据库连接。
        """
        if self.connection:
            self.connection.close()
            print("数据库连接已关闭。")

    def insert_metrics(self, folder_path, metrics_data, metrics_id):
        """
        将指标数据插入到 metrics_TTC 表中。

        :param folder_path: 包含 eval_results 目录的文件夹路径。
        :param metrics_data: 代表指标数据的二维列表。
        :param metrics_id: 关联的 metrics 表的 id。
        """
        try:
            # TODO:UPDATE12.20：长度不一定为10,没有的设置为空
            # 确保 metrics_data 是一个二维列表
            if not isinstance(metrics_data, list):
                print(f"无效-insert_metrics: metrics_data 不是一个列表")
                return

            # 如果 metrics_data 的长度小于 10，则填充空列表直到长度为 10
            while len(metrics_data) < 10:
                metrics_data.append([])
            # 替换 inf 和 -inf 为 0
            # TODO：UPDATE12.23：但是这个点要空着，不能用后面的直接覆盖掉。不能是inf，否则无法插入数据库。解决办法：改为特殊值-1024
            metrics_data = [[-1024 if math.isinf(x) else x for x in sublist] for sublist in metrics_data]

            with self.connection.cursor() as cursor:
                # 动态生成列名和占位符
                columns = ', '.join([f'attribute{i + 1}' for i in range(len(metrics_data))]) + ', metrics_id'
                placeholders = ', '.join(['%s'] * (len(metrics_data) + 1))

                # 构造 SQL 查询
                sql = f"""
                INSERT INTO metrics_TTC ({columns})
                VALUES ({placeholders})
                """

                # 准备要插入的值
                values = [json.dumps(sublist) for sublist in metrics_data] + [metrics_id]

                # 使用提供的值执行 SQL 查询
                cursor.execute(sql, tuple(values))
                self.connection.commit()
                print(f"已插入 metrics_id 为 {metrics_id} 的指标数据。")
        except pymysql.MySQLError as e:
            print(f"插入指标数据时出错: {e}")
            self.connection.rollback()

    def find_metrics_id_by_folder_name(self, folder_name):
        """
        根据文件夹名称查找对应的 metrics 表中的记录 id。

        :param folder_name: 文件夹名称。
        :return: 对应的 metrics 表中的记录 id，如果找不到则返回 None。
        """
        try:
            with self.connection.cursor() as cursor:
                sql = "SELECT id FROM metrics WHERE folder_name = %s"
                cursor.execute(sql, (folder_name,))
                result = cursor.fetchone()
                if result:
                    return result['id']
                else:
                    print(f"未找到对应的 metrics 记录: {folder_name}")
                    return None
        except Exception as e:
            print(f"读取或查找 metrics_id 时出错: {e}")
            return None

    def check_metrics_id_exists_in_metrics_ttc(self, metrics_id):
        """
        检查 metrics_TTC 表中是否存在指定的 metrics_id。

        :param metrics_id: 要检查的 metrics_id。
        :return: 如果存在返回 True，否则返回 False。
        """
        try:
            with self.connection.cursor() as cursor:
                sql = "SELECT COUNT(*) AS count FROM metrics_TTC WHERE metrics_id = %s"
                cursor.execute(sql, (metrics_id,))
                result = cursor.fetchone()
                return result['count'] > 0
        except pymysql.MySQLError as e:
            print(f"检查 metrics_id 存在性时出错: {e}")
            return False

    def insert_metrics_from_folder(self, base_path):
        """
        读取子文件夹中的 records_ttc.pkl 文件并将指标数据插入到数据库中。

        :param base_path: log/exp 目录的基本路径。
        """
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                pkl_file_path = os.path.join(folder_path, "eval_results", "records_ttc.pkl")

                if os.path.isfile(pkl_file_path):
                    try:
                        with open(pkl_file_path, 'rb') as f:
                            metrics_data_list = pickle.load(f)

                        # 查找对应的 metrics_id
                        metrics_id = self.find_metrics_id_by_folder_name(folder_name)
                        if metrics_id is None:
                            print(f"未找到对应的 metrics_id, 跳过插入: {folder_path}")
                            continue


                        # 检查 metrics_id 是否已经在 metrics_TTC 表中存在
                        if self.check_metrics_id_exists_in_metrics_ttc(metrics_id):
                            print(f"metrics_id {metrics_id} 已经存在于 metrics_TTC 表中, 跳过插入: {folder_path}")
                            continue

                        # 插入 metrics 数据
                        # for data in metrics_data_list:
                        #     self.insert_metrics(folder_path, data, metrics_id)
                        self.insert_metrics(folder_path, metrics_data_list, metrics_id)#TODO: 需要保证每一个metrics_id唯一。

                    except Exception as e:
                        print(f"读取文件 {pkl_file_path} 时出错: {e}")


# 示例用法
if __name__ == "__main__":
    inserter = MetricsTTCInserter(host='localhost', user='heihuhu', password='1234', db='db')
    inserter.connect()
    inserter.insert_metrics_from_folder('../log/exp/')
    inserter.close()