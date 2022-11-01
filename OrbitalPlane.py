# OrbitalPlane defines the presence of satellites (clients) on a single plane with similar entropy and regulates
# acceptance of a new satellite or removal of a satellite from the plane.
# Note: Terminology- Orbit and Orbital plane are the same. Orbital plane has been used to avoid any confusions.

import numpy as np
from time import time, sleep
import threading
import asyncio
import uuid
import requests
import multiprocessing


def circle(list_name):
    while True:
        for connection in list_name:
            yield connection


class OrbitalPlane:

    def __init__(self, planet, entropy, azimuth, orbital_system, local_stages, id=None):
        self.Azimuth = azimuth  # frequency at which the update takes place.
        self.AzimuthThreshold = 100
        if id is not None:
            self.OrbitalID = id
        else:
            self.OrbitalID = uuid.uuid4()
        self.LocalSatellites = {}
        self.LocalEntropy = entropy
        self.Planet = planet
        self.LocalStages = local_stages
        if self.LocalStages:
            self.ActiveStage = self.LocalStages[-1]
        else:
            self.ActiveStage = 0
        self.TimeInterval = 3 # 86400  # seconds
        self.Threshold = 50000000.0
        self.Restrictions = True  # Placeholder for satellites to provide framework of accepting a new satellite
        self.OrbitalSystem = orbital_system
        self.ProcessThread = []
        # loop = asyncio.get_event_loop()
        # task = loop.create_task(self.maintain_orbit())

        # self.maintain_orbit()
    def start_orbit(self):
        self.ProcessThread = threading.Thread(name='PMO_'+str(self.OrbitalID), target=self.maintain_orbit)
        self.ProcessThread.start()

    def add_satellite(self, satellite):
        self.LocalSatellites[satellite.SatelliteID] = satellite

    def remove_satellite(self, satellite_id):
        del self.LocalSatellites[satellite_id]

    def close_orbit(self):
        del self

    def satellite_reentry(self, satellite_id):
        status = self.OrbitalSystem.signal_removal(satellite_id)
        if status:
            del self.LocalSatellites[satellite_id]
        else:
            print('Re-entry failed at orbital control system.')

    def evaluate_entropy(self):
        ent_dict = {}
        ent_array = np.zeros(len(self.LocalSatellites.keys()))
        for i, s in enumerate(self.LocalSatellites.keys()):
            go = self.LocalSatellites[s].get_entropy('1.0')
            if go.status_code == 200:
                ent_dict[s] = str(go.text)
                ent_array[i] = str(go.text)
            else:
                ent_dict[s] = 5000
                ent_array[i] = 5000     # TODO: test required.
        return np.mean(ent_array), self.max_deviation(ent_array), ent_dict

    def enter_orbit(self, satellite):
        if self.Restrictions:
            self.LocalSatellites[int(satellite.SatelliteID)] = satellite
            return True

    def launch_rocket(self):
        while True:
            try:
                readiness = requests.get(self.Planet.ContainerURL+'/ready_state')
                print(readiness.content.decode('ascii'))
                break
            except:
                continue
        if readiness.content.decode('ascii') == "\"True\"":
            for kl, ks in enumerate(list(self.LocalSatellites.keys())):
                self.LocalSatellites[ks].request_launch(1.0)
                self.LocalSatellites[ks].request_extras(1.0)
        else:
            print("error in model launch")

    # This will actively maintain orbit at a particular mean entropy and remove satellite with higher or lower
    # entropies. Before removal of any satellite, a request shall be sent to the orbital system manager, which
    # will allocate a new orbital plane or assign an existing orbital plane to the removed satellite.
    # No active satellite is to be left out of orbital system.
    def maintain_orbit(self):
        counter_loop = 1
        check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
        while True:
            # print(self.OrbitalSystem.StartFlag)
            if self.OrbitalSystem.StartFlag:
                # print('test 1')
                # sleep(self.TimeInterval - time() % self.TimeInterval)

                for kl, ks in enumerate(list(self.LocalSatellites.keys())):
                    if str(self.LocalSatellites[ks].get_rtj())[1:-1] == str("BBlink"):
                        # if check_rtj[kl] != 1:
                        #     print(ks)
                        #     print(len(list(self.LocalSatellites.keys())))
                        #     print('getting_rtj')
                        #     self.LocalSatellites[ks].time_the_update()
                        check_rtj[kl] = 1
                    # print(str(self.LocalSatellites[ks].get_rtj()[1:-1]))
                    # print(str(kl)+":"+str(check_rtj[kl]))
                if len(list(self.LocalSatellites.keys()))<1:
                    continue
                if not np.all(check_rtj):
                    continue
                check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
                lock = threading.Lock()
                lock.acquire()
                print(str(self.OrbitalID)+' Loop_count:' + str(counter_loop))
                lock.release()
                for thread in threading.enumerate():
                    print(thread.name)
                    print(thread.is_alive())
                l_list = list(self.LocalSatellites.keys())
                lr_list = circle(l_list)
                for ns in list(self.LocalSatellites.keys()):
                    self.LocalSatellites[ns].reenter_stages(1.0, 'ss', 's_stage')
                now_sat = 0
                for ns in range(len(list(self.LocalSatellites.keys()))):
                    if ns == 0:
                        now_sat = next(lr_list)
                    next_sat = next(lr_list)
                    self.LocalSatellites[now_sat].transfer_stages(self.LocalSatellites[next_sat], 1.0)
                    # This for loop is set for aux stage upload (ensemble learning)
                    for r, der in enumerate(list(self.LocalSatellites.keys())):
                        if der == now_sat:
                            # print('der darrred')
                            continue
                        self.LocalSatellites[now_sat].request_aux_stages(self.LocalSatellites[der], 1.0, stage_id=str(r)
                                                                         , stage_level='s_stage')
                    # TODO: change to planet vol
                    rtj_flag = {'RTJ': 'False'}

                    self.LocalSatellites[now_sat].set_rtj(rtj_flag)
                    now_sat = next_sat
                ent_measure, ent_dev, ent_dict = self.evaluate_entropy()
                for nq in list(self.LocalSatellites.keys()):
                    ou = self.LocalSatellites[nq].get_azimuth()
                    lock = threading.Lock()
                    lock.acquire()
                    print('***********************')
                    print(str(self.OrbitalID))
                    print('Azimuth times')
                    print(self.LocalSatellites[nq].Azimuth_local)
                    print('***********************')
                    lock.release()

                    if abs(self.LocalSatellites[nq].Azimuth_local - self.Azimuth) > self.AzimuthThreshold:
                        self.LocalSatellites[nq].StatelessReason = 'azimuth'
                        asyncio.run(self.OrbitalSystem.signal_displacement(nq))
                        print(self.OrbitalSystem.StatelessSatellites)
                        del self.LocalSatellites[nq]
                        check_rtj = np.zeros(len(list(self.LocalSatellites.keys())))
                azi = 0.0
                for nl in list(self.LocalSatellites.keys()):
                    azi += float(self.LocalSatellites[nl].Azimuth_local)
                lock = threading.Lock()
                lock.acquire()
                print('-----------------------')
                print(str(self.OrbitalID))
                print('testing beta')
                print(len(list(self.LocalSatellites.keys())))
                print('-----------------------')
                lock.release()
                if len(list(self.LocalSatellites.keys())) > 0:
                    self.Azimuth = azi/len(list(self.LocalSatellites.keys()))
                # else:
                    # self.OrbitalSystem.signal_empty_orbit(self)
                    # self.ProcessThread.join()
                # if (ent_dev < abs(self.Threshold - ent_measure)) or (self.Threshold + ent_measure) > ent_dev:
                #     print('inner sanctum')
                #     temp_list = list(ent_dict.values())
                #     temp_list_2 = temp_list - ent_measure
                #     temp_max = max(temp_list_2)
                #     temp_idx = temp_list_2.index(temp_max)
                #     rem_id = list(ent_dict.keys())[list(ent_dict.values()).index(temp_list[temp_idx])]
                #     status = self.OrbitalSystem.signal_displacement(rem_id, (temp_max + ent_measure))
                #     if status:
                #         del self.LocalSatellites[rem_id]
                self.LocalEntropy = ent_measure
                self.OrbitalSystem.update_orbital_status(self.OrbitalID, self.LocalEntropy)
                counter_loop += 1

    @staticmethod
    def max_deviation(a):
        avg = sum(a, 0.0) / len(a)
        max_dev = max(abs(el - avg) for el in a)
        return max_dev

