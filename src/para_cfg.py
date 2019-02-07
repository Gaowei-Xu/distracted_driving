import ConfigParser

class ParaCfg:
    def __init__(self, cfgfile):
        config = ConfigParser.ConfigParser()
        config.read(cfgfile)

        self.INPUT_TYPE = config.get('BaseConfigure', 'INPUT_TYPE')
        self.FRAME_WIDTH = config.getint('BaseConfigure', 'FRAME_WIDTH_INT')
        self.FRAME_HEIGHT = config.getint('BaseConfigure', 'FRAME_HEIGHT_INT')
        self.STRIDE = config.getint('BaseConfigure', 'STRIDE_INT')
        self.GENERATE_RAW_FRAME = config.getboolean('BaseConfigure', 'GENERATE_RAW_FRAME_BOOLEAN')
        self.GENERATE_FRAME_WITH_PROB = config.getboolean('BaseConfigure', 'GENERATE_FRAME_WITH_PROB_BOOLEAN')
        self.GENERATE_LOGFILE = config.getboolean('BaseConfigure', 'GENERATE_LOGFILE_BOOLEAN')
        self.SHOW_FRAME_REALTIME = config.getboolean('BaseConfigure', 'SHOW_FRAME_REALTIME_BOOLEAN')
        self.NET_BASE_PATH = config.get('PathConfigure', 'NET_BASE_PATH')
        self.MODEL_PATH = config.get('PathConfigure', 'MODEL_PATH')
        self.DEPLOY_PATH = config.get('PathConfigure', 'DEPLOY_PATH')
        self.MEAN_BINARY_PATH = config.get('PathConfigure', 'MEAN_BINARY_PATH')
        self.IMAGE_SET_BASE_PATH = config.get('PathConfigure', 'IMAGE_SET_BASE_PATH')
        self.VIDEO_PATH = config.get('PathConfigure', 'VIDEO_PATH')
        self.LOGFILE_PATH = config.get('PathConfigure', 'LOGFILE_PATH')
        self.OUTPUT_FRAMES_PATH = config.get('PathConfigure', 'OUTPUT_FRAMES_PATH')
        self.OUTPUT_FRAMES_WITH_PROB_PATH = config.get('PathConfigure', 'OUTPUT_FRAMES_WITH_PROB_PATH')

    def to_log(self, logger):
        logger.info('INPUT_TYPE = ' + str(self.INPUT_TYPE))
        if self.INPUT_TYPE == 0 or self.INPUT_TYPE == 1 or self.INPUT_TYPE == 2:
            pass
        else:
            logger.error('INPUT_TYPE should be 0, 1 or 2, please check the configure file.')
        logger.info('FRAME_WIDTH = ' + str(self.FRAME_WIDTH))
        logger.info('FRAME_HEIGHT = ' + str(self.FRAME_HEIGHT))
        logger.info('STRIDE = ' + str(self.STRIDE))
        logger.info('GENERATE_RAW_FRAME = ' + str(self.GENERATE_RAW_FRAME))
        logger.info('GENERATE_FRAME_WITH_PROB = ' + str(self.GENERATE_FRAME_WITH_PROB))
        logger.info('GENERATE_LOGFILE_BOOLEAN = ' + str(self.GENERATE_LOGFILE))
        logger.info('SHOW_FRAME_REALTIME = ' + str(self.SHOW_FRAME_REALTIME))
        logger.info('NET_BASE_PATH = ' + str(self.NET_BASE_PATH))
        logger.info('MODEL_PATH = ' + str(self.MODEL_PATH))
        logger.info('DEPLOY_PATH = ' + str(self.DEPLOY_PATH))
        logger.info('MEAN_BINARY_PATH = ' + str(self.MEAN_BINARY_PATH))
        logger.info('IMAGE_SET_BASE_PATH = ' + str(self.IMAGE_SET_BASE_PATH))
        logger.info('VIDEO_PATH = ' + str(self.VIDEO_PATH))
        logger.info('LOGFILE_PATH = ' + str(self.LOGFILE_PATH))
        logger.info('OUTPUT_FRAMES_PATH = ' + str(self.OUTPUT_FRAMES_PATH))
        logger.info('OUTPUT_FRAMES_WITH_PROB_PATH = ' + str(self.OUTPUT_FRAMES_WITH_PROB_PATH))
