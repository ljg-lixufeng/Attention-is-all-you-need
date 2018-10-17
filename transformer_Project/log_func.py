import logging
import datetime
import os


def my_log(logname):
    '''
    :param name: 申请logger的名字
    :return:申请号的logger

    函数用来申请logger，同时返回申请的logger；
    申请logger：logger_name = test_logging.my_log('name')
    调用logger：logger_name.debug('mylog test b: %d'%(b))
    '''

    today = datetime.date.today()

    if not os.path.exists('log/'):
        os.mkdir('log/')

    logfile = 'log/' + str(today)
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(name)s: %(levelname)s: %(message)s')
    #输出到控制台
    hd = logging.StreamHandler()
    hd.setFormatter(formatter)
    #输出到文本
    fh = logging.FileHandler(logfile)
    fh.setFormatter(formatter)
    #增加handler
    logger.addHandler(fh)
    logger.addHandler(hd)
    return logger