"""Scalar feedback from completed decoder-required bytes, never image metrics."""
from dataclasses import asdict,dataclass
import math


@dataclass
class RateBudgetState:
    frames:int=0
    total_bytes:int=0
    log_multiplier:float=0.
    last_frame:int=0
    last_bytes:int=0
    last_multiplier:float=1.

    @property
    def multiplier(self):
        return math.exp(self.log_multiplier)

    def capture(self):
        return asdict(self)

    @classmethod
    def restore(cls,packet,maximum):
        state=cls() if packet is None else cls(**packet)
        for name in ('frames','total_bytes','last_frame','last_bytes'):
            value=getattr(state,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ValueError(f'Invalid budget {name}')
        if (state.frames!=state.last_frame or state.last_bytes>state.total_bytes or
                (state.frames==0 and (state.total_bytes or state.last_bytes)) or
                (state.frames>0 and (state.last_bytes==0 or state.total_bytes==0))):
            raise ValueError('Budget history does not match completed frames')
        if (not math.isfinite(state.log_multiplier) or not 0<=state.log_multiplier<=math.log(maximum)+1e-12 or
                not math.isfinite(state.last_multiplier) or state.last_multiplier<1 or
                (state.last_multiplier>maximum and not math.isclose(state.last_multiplier,maximum,rel_tol=1e-12))):
            raise ValueError('Invalid budget multiplier history')
        return state

    def update(self,frame,complete_bytes,target,gain,maximum):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in (frame,complete_bytes,target)):
            raise ValueError('Budget updates require a dynamic frame and positive byte counts')
        if not math.isfinite(gain) or gain<=0 or not math.isfinite(maximum) or maximum<1:
            raise ValueError('Invalid budget feedback bounds')
        if frame==self.last_frame:
            if complete_bytes!=self.last_bytes:
                raise ValueError('Repeated budget update changed its encoded byte count')
            used=self.last_multiplier
        else:
            if frame!=self.last_frame+1:
                raise ValueError('Budget feedback must follow each completed dynamic frame exactly once')
            used=self.multiplier
            self.log_multiplier=min(math.log(maximum),max(0.,
                self.log_multiplier+gain*math.log(complete_bytes/target)))
            self.frames+=1;self.total_bytes+=complete_bytes
            self.last_frame=frame;self.last_bytes=complete_bytes;self.last_multiplier=used
        return dict(frame=frame,target_bytes=target,complete_bytes=complete_bytes,
            used_multiplier=used,next_multiplier=self.multiplier,
            mean_complete_bytes=self.total_bytes/self.frames,
            feedback='previous completed bytes only; applied next frame')
