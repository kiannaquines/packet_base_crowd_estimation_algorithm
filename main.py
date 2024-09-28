from utilities.packet_base_estimation_algorithm import PackBaseEstimationAlgorithm
from utilities.database import Database

if __name__ == "__main__":
    database = Database()
    algorithm = PackBaseEstimationAlgorithm()
    data = database.fetch_data()
    algorithm.load_data(data)
    algorithm.run()
