from simulation.simulation import MTJSimulation
import numpy as np

class DeviceSweep:
    def __init__(self, device_classes, param_sets):
        self.device_classes = device_classes
        self.param_sets = param_sets

    def run_all(self, j_stt_arr, j_she, flips):
        results = []
        for dev_class in self.device_classes:
            for ps in self.param_sets:
                bitstream_averages = []
                sim_out = []
                for j_stt in j_stt_arr:
                    dev = dev_class(ps)
                    sim = MTJSimulation(dev)
                    res = sim.run((np.pi/2, 0, 0, 0, 0), j_she=j_she, j_stt=j_stt, flips=flips)
                    sim_out.append({
                        "device": dev_class.__name__,
                        "params": ps,
                        "j_stt": j_stt,
                        "result": res
                    })
                    bitstream_averages.append(np.mean(res["bitstream"]))
                results.append({"simulation_output": sim_out, "bitstream_averages": bitstream_averages, "j_stt_arr": j_stt_arr})
        return results