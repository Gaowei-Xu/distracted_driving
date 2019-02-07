import ConfigParser

config = ConfigParser.ConfigParser()
config.read('appConfig.cfg')


an_int = config.getint('BaseConfigure', 'INPUTTYPE_int')
print an_int


GENERATE_RAW_FRAME = config.getboolean('BaseConfigure', 'GENERATE_RAW_FRAME_bool')

GENERATE_FRAME_WITH_PROB_bool = config.getboolean('BaseConfigure', 'GENERATE_FRAME_WITH_PROB_bool')

print GENERATE_RAW_FRAME

print GENERATE_FRAME_WITH_PROB_bool





net_base_path = config.get('PathConfigure', 'NET_BASE_PATH')
print net_base_path
