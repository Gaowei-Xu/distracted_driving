from time import gmtime, strftime
import os
import logging


class Logger:
    def __init__(self, LOGFILE_PATH):
        self.__logfilename = strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        self.__logfilefullpath = os.path.join(LOGFILE_PATH, self.__logfilename)
        logging.basicConfig(level = logging.DEBUG,
                            format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt = '%a, %d %b %Y %H:%M:%S',
                            filename = self.__logfilefullpath,
                            filemode='w')
        self.__logger = logging.getLogger(self.__logfilefullpath)
        return

    def get_logger(self):
        return self.__logger


















