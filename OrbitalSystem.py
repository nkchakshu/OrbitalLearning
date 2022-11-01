# Developed by Neeraj Kavan Chakshu in 2022.

# This is the Orbital Learning Project. A combination of Federated split learning, cyclic weight transfer,
# swarm learning and reinforcement learning is used.

# Class: Orbital system - The core section of orbital learning where the planet-satellite system is defined
# for a network.

# Note: Terminology - client and server are used only in this file to assist end users. In all files, planet
# refers to server and satellite refers to client.

import RocketFactory
import asyncio
from time import sleep, time
import operator
import OrbitalPlane
import Satellite
from sklearn.neighbors import BallTree
import threading
import numpy as np
import copy


class OrbitalSystem:

    def __init__(self, clients, servers=None):
        if servers is None:
            servers = {}
        self.Satellites = {} # Type: Dictionary
        self.Planets = servers
        self.TimeInterval = 20  # seconds
        self.Threshold = 10000000000.0
        self.AzimuthThreshold = 100
        self.Entropies = {}
        self.Azimuths = {}
        self.StatelessSatellites = []
        self.OrbitalPlaneAltitudes = {}  # Not to be confused with entropies, this shows only the order (int) of planes.
        self.SignalSS = False  # Signal Stateless Satellite
        self.AzimuthTPThreshold = 5 * 36000   # The range for difference in time period (in seconds) for any
        # particular azimuth.
        self.StartFlag = False
        print('ddd')
        if servers == {}:
            self.Planets = RocketFactory.BuildDefaults.generate_planet()
        self.OrbitalPlanes = RocketFactory.BuildDefaults.initialise_orbital_planes(self, self.Planets['def'])
        print('fdg')
        c_sats = {}
        for n in range(clients):
            c_sats[n] = Satellite.Satellite(initial_entropy=1.0, port=4000+n, orbit=1.0, satellite_id=int(n),
                                orbital_plane=self.OrbitalPlanes['def'], vol_id=str(n))
            print('hhop')
        self.StartFlag = True
        self.Satellites = c_sats.copy()
        self.OrbitalPlanes['def'].LocalSatellites = self.Satellites.copy()
        self.OrbitalPlanes['def'].launch_rocket()
        # self.OrbitalPlanes['def'].maintain_orbit()
        self.OrbitalPlanes['def'].start_orbit()
        # self.active_control()
        background_process = threading.Thread(name='PAC_OrbitalSystem', target=self.active_control)
        background_process.start()
        # self.default_server()

    def __getstate__(self):
        print('Number of Clients:' + str(len(list(self.Satellites))))
        print('Number of Servers:' + str(len(self.Planets)))
        print('Number of Orbital Planes:' + str(len(list(self.OrbitalPlanes))))
        print('Max Entropy:' + str(max(list(self.Entropies.values()))))
        print('Min Entropy:' + str(min(list(self.Entropies.values()))))

    def add_client(self, client):
        self.Satellites.append(client)

    def add_server(self, server):
        self.Planets.append(server)

    # default_server will be used to create a server container, if a server module is not defined/provided.
    def default_server(self):
        if not self.Planets:
            dev_server = RocketFactory.BuildDefaults.generate_planet()
            self.Planets['0'] = dev_server

    def multi_planet_system(self):
        pass

    def signal_empty_orbit(self, orbit):
        lock = threading.Lock()
        lock.acquire()
        print('empty orbits check')
        print(orbit.OrbitalID)
        print(list(self.OrbitalPlanes.keys()))
        print(list(orbit.LocalSatellites.keys()))
        lock.release()

        del self.OrbitalPlanes[orbit.OrbitalID]
        del self.Entropies[orbit.OrbitalID]
        del self.Azimuths[orbit.OrbitalID]
        del self.OrbitalPlaneAltitudes[orbit.OrbitalID]

    # This is to signal that an active satellite is stateless as it has been removed from an orbit.
    async def signal_displacement(self, satellite_id):
        lock = threading.Lock()
        lock.acquire()
        print('displacemnt check')
        print(self.StatelessSatellites)
        print(list(self.Satellites.keys()))
        lock.release()
        # temp_dict = self.Satellites.copy()
        self.StatelessSatellites.append(int(satellite_id))
        self.SignalSS = True
        return True

    # This is to signal removal a satellite from the entire system as it has
    # re-entered (disconnects due to maintenance / leaves the system)
    def signal_removal(self, satellite_id):
        del self.Satellites[satellite_id]
        if satellite_id in self.StatelessSatellites:
            del self.StatelessSatellites[satellite_id]
        return True

    def update_orbital_status(self, orbital_id, orbital_entropy):
        self.Entropies[orbital_id] = orbital_entropy

    # This defines the controlling of different orbital planes and assigning satellites.
    def active_control(self):

        start_time = time()
        while True:
            if self.StartFlag:
                sleep(self.TimeInterval - time() % self.TimeInterval)
                # print('Time elapsed: ' + str(time()-start_time))

                for plane in list(self.OrbitalPlanes.keys()):
                    self.Entropies[self.OrbitalPlanes[plane].OrbitalID] = self.OrbitalPlanes[plane].LocalEntropy
                res_key_ar = [x for x in list(self.Entropies.keys())]
                res_key_ar = np.array(res_key_ar)
                rf = res_key_ar[0]
                for plane in list(self.OrbitalPlanes.keys()):
                    self.Azimuths[self.OrbitalPlanes[plane].OrbitalID] = self.OrbitalPlanes[plane].Azimuth

                # Check required
                ent_sorted = dict(sorted(self.Entropies.items(), key=operator.itemgetter(1))).keys()
                for i, u in enumerate(ent_sorted):
                    self.OrbitalPlaneAltitudes[u] = i

                # If a satellites exist without an orbit, since they have been removed by an orbital plane.
                if self.SignalSS:
                    for sats in self.StatelessSatellites:
                        # if self.StatelessSatellites[sats].StatelessReason == 'entropy':
                        #     ent = self.StatelessSatellites[sats].get_entropy()
                        #     res_key, res_val = min(self.Entropies.items(), key=lambda x: abs(ent - x[1]))
                        #     if ent < 1.0:
                        #         print('error in entropy (less than 1.0)')
                        #     if abs(res_val - ent) <= self.Threshold:
                        #         status = self.OrbitalPlanes[res_key].enter_orbit(self.StatelessSatellites[sats])
                        #         if status:
                        #             self.StatelessSatellites[sats].SharedStages = self.OrbitalPlanes[res_key].LocalStages
                        #             del self.StatelessSatellites[sats]
                        #         # This else is assuming orbital planes have decision to reject a new satellite,
                        #         # for example, if the satellites vote not to allow.
                        #         else:
                        #             stage_nos = self.StatelessSatellites[sats].SharedStages
                        #             new_orbit = OrbitalPlane.OrbitalPlane(self.Planets['def'], ent, 500, self, stage_nos)
                        #             self.OrbitalPlanes[new_orbit.OrbitalID] = new_orbit
                        #             stats = new_orbit.enter_orbit(self.StatelessSatellites[sats])
                        #             if stats:
                        #                 del self.StatelessSatellites[sats]
                        #                 print('satellite placed in new orbit')
                        #     else:
                        #         stage_nos = self.StatelessSatellites[sats].SharedStages
                        #         new_orbit = OrbitalPlane.OrbitalPlane(self.Planets['def'], ent, 500, self, stage_nos)
                        #         self.OrbitalPlanes[new_orbit.OrbitalID] = new_orbit
                        #         stats = new_orbit.enter_orbit(self.StatelessSatellites[sats])
                        #         if stats:
                        #             del self.StatelessSatellites[sats]
                        #             print('satellite placed in new orbit')
                        # elif self.StatelessSatellites[sats].StatelessReason == 'azimuth':
                        ent = self.Satellites[sats].get_entropy('1.0')
                        azim = self.Satellites[sats].Azimuth_local
                        bhk = np.zeros((len(list(self.Entropies.items())), 2))
                        bhk[:, 0] = np.fromiter(self.Entropies.values(), dtype=float)
                        bhk[:, 1] = np.fromiter(self.Azimuths.values(), dtype=float)
                        tree = BallTree(bhk, leaf_size=2)
                        dist, res_key = tree.query([[float(ent.content), azim]], k=1)
                        res_val_ent = bhk[res_key, 0]
                        res_val_azimuth = bhk[res_key, 1]
                        res_key_ar = [x for x in list(self.Entropies.keys())]
                        res_key_ar = np.array(res_key_ar)
                        if float(ent.content) < 1.0:
                            print('error in entropy (less than 1.0)')
                        if (abs(res_val_ent - float(ent.content)) <= self.Threshold) and (abs(res_val_azimuth - azim) <= self.AzimuthThreshold):
                            res_id = res_key_ar[int(res_key)]
                            print('reskey check')
                            print(res_key)
                            status = self.OrbitalPlanes[res_id].enter_orbit(self.Satellites[sats])
                            if status:
                                self.Satellites[sats].SharedStages = self.OrbitalPlanes[
                                    res_id].LocalStages
                                del self.Satellites[sats]
                            # This else is assuming orbital planes have decision to reject a new satellite,
                            # for example, if the satellites vote not to allow.
                            else:
                                stage_nos = self.Satellites[sats].SharedStages
                                new_orbit = OrbitalPlane.OrbitalPlane(self.Planets['def'], float(ent.content),
                                                                      self.Satellites[sats].Azimuth_local,
                                                                      self, stage_nos)
                                new_orbit.start_orbit()
                                self.OrbitalPlanes[new_orbit.OrbitalID] = new_orbit
                                stats = new_orbit.enter_orbit(self.Satellites[sats])
                                if stats:
                                    # self.StatelessSatellites.pop(sats)
                                    print('satellite placed in new orbit')
                        else:
                            stage_nos = self.Satellites[sats].SharedStages
                            new_orbit = OrbitalPlane.OrbitalPlane(self.Planets['def'], float(ent.content),
                                                                  self.Satellites[sats].Azimuth_local,
                                                                  self, stage_nos)
                            new_orbit.start_orbit()
                            self.OrbitalPlanes[new_orbit.OrbitalID] = new_orbit
                            stats = new_orbit.enter_orbit(self.Satellites[sats])
                            if stats:
                                # self.StatelessSatellites.pop(sats)
                                print('satellite placed in new orbit')
                self.StatelessSatellites = []
            if abs(time()-start_time)>2700.0:
                break
                # To find orbital planes with just one satellite and see if those satellites can join other orbits.
                # for plane in list(self.OrbitalPlanes.keys()):
                #     if len(list(self.OrbitalPlanes[plane].LocalSatellites.keys())) == 1:
                #         sat_id = list(self.OrbitalPlanes[plane].LocalSatellites.keys())[0]
                #         ent_local = self.OrbitalPlanes[plane].LocalSatellites[sat_id].IndividualEntropy
                #         self.Entropies[self.OrbitalPlanes[plane].OrbitalID] = -300.0
                #         res_key, res_val = min(self.Entropies.items(), key=lambda x: abs(ent_local - x[1]))
                #         if ent_local < 1.0:
                #             print('error in entropy (less than 1.0)')
                #         if abs(res_val - ent_local) <= self.Threshold:
                #             status = self.OrbitalPlanes[res_key].enter_orbit(self.Satellites[sat_id])
                #             if status:
                #                 flag = self.OrbitalPlanes[plane].close_plane()
                #                 del self.OrbitalPlanes[plane]
