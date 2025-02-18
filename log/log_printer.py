import pickle
import mysql.connector
from mysql.connector import Error


def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    # 如果数据是一个单一的字典，将其转换为字典列表
    if isinstance(data, dict):
        data = [data]

    return data


def insert_data_to_mysql(data, host_name, user_name, user_password, db_name):
    connection = None  # 初始化连接为None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        cursor = connection.cursor()

        for entry in data:
            if isinstance(entry, dict):  # 确保 entry 是字典
                query = """
                INSERT INTO metrics (
                    collision_rate, avg_red_light_freq, avg_stop_sign_freq, 
                    out_of_road_length, route_following_stability, route_completion, 
                    avg_time_spent, avg_acceleration, avg_yaw_velocity, 
                    avg_lane_invasion_freq, safety_os, task_os, comfort_os, final_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = tuple(entry.values())
                cursor.execute(query, values)
            else:
                print(f"Skipping non-dict entry: {entry}")

        connection.commit()
        print("Data inserted successfully.")

    except Error as e:
        print(f"The error '{e}' occurred")

    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == "__main__":
    pkl_file_path = './exp/exp_basic_Generate_Traffic_seed_100/eval_results/records_ttc.pkl'  # 替换为你的 .pkl 文件路径
    host = "localhost"
    user = "root"
    password = "123"  # 替换为你的 MySQL root 密码
    database = "db"

    data = load_data_from_pkl(pkl_file_path)
    print(data)  # 打印加载的数据以进行验证
#    insert_data_to_mysql(data, host, user, password, database)



