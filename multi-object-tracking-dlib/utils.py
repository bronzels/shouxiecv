import datetime

class FPS:
    def __init__(self):
        # store the start time, end time, total number of all frames
        # that are examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
    
    def start(self):
        self._start = datetime.datetime.now()
        return self
    
    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames during
        # the start and end intervals
        self._numFrames += 1
    
    def elapsed(self):
        # return the total number of frames between the start and
        # end interval
        return (self._end - self._start).total_seconds()
    
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()
    