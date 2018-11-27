import threading
import time

#def main():
#    print(threading.active_count())
#    print(threading.enumerate()) # see the thread list
#    print(threading.current_thread())

def thread_job():
    print('T1 start\n')
    for i in range(10):
        time.sleep(0.1)
    print('T1 finish\n')

def T2_job():
    print('T2 start\n')
    print('T2 finish\n')

def main():
    thread1 = threading.Thread(target=thread_job, name='T1')
    thread2 = threading.Thread(target=T2_job, name='T2')
    thread1.start()
    thread2.start()
    thread2.join()
    thread1.join()
    print('all done\n')
    
if __name__ == '__main__':
    main()
