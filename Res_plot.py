import cvxpy as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

class Res_plot(object):
    def __init__(self,env):
        super(Res_plot, self).__init__()
        self.env = env
        self._build_result()

    def _build_result(self):
        return

    def plot_UAV_GT (self,w_k, UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,N_slot, slot_ris,slot_ris_no_shift,slot_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris):
        myfont = matplotlib.font_manager.FontProperties(
        fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
        t_ris = np.zeros((1, self.env.eps), dtype=np.int)
        t_ris_no_shift = np.zeros((1, self.env.eps), dtype=np.int)
        t_no_ris = np.zeros((1, self.env.eps), dtype=np.int)

        for e in range(self.env.eps):
            t_ris[0, e] = np.sum(UAV_flight_time_ris[e, :])/10
            t_ris_no_shift[0, e] = np.sum(UAV_flight_time_ris_no_shift[e, :])/10
            t_no_ris[0, e] = np.sum(UAV_flight_time_no_ris[e, :])/10

        average = np.sum(t_ris[0,:])+np.sum(t_ris_no_shift[0,:])+np.sum(t_no_ris[0,:])
        average = (average/3/self.env.eps)*1.5

        #name_list = ['episode 1', 'episode 2', 'C', 'D']
        x = list(range(0,self.env.eps,6))
        total_width, n = 3, 3
        width = total_width / n
        plt.ylim(0,average)
        plt.bar(x, t_ris[0,0:3:self.env.eps], width=width, label='RIS-Assisted UAV', fc='g')
        for i in range(len(x)):
            x[i] += width
        plt.bar(x, t_ris_no_shift[0,0:3:self.env.eps], width=width, label='UAV-R/P', fc='b')
        for i in range(len(x)):
            x[i] += width
        plt.bar(x, t_no_ris[0,0:3:self.env.eps], width=width, label='UAV/R', fc='r')
        plt.legend()
        plt.show()

        for e in range((self.env.eps-10),self.env.eps):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            x_ris = []
            y_ris = []
            z_ris = []

            x_ris_no_shift = []
            y_ris_no_shift  = []
            z_ris_no_shift  = []

            x_no_ris= []
            y_no_ris = []
            z_no_ris = []

            for slot in range(N_slot):
                if slot < slot_ris[0,e]:
                    x_ris.append(UAV_trajectory_ris[e,slot,0])
                    y_ris.append(UAV_trajectory_ris[e,slot,1])
                    z_ris.append(UAV_trajectory_ris[e,slot,2])
                if slot< slot_ris_no_shift[0,e]:
                    x_ris_no_shift.append(UAV_trajectory_ris_no_shift[e, slot, 0])
                    y_ris_no_shift.append(UAV_trajectory_ris_no_shift[e, slot, 1])
                    z_ris_no_shift.append(UAV_trajectory_ris_no_shift[e, slot, 2])
                if slot < slot_no_ris[0,e]:
                    x_no_ris.append(UAV_trajectory_no_ris[e,slot, 0])
                    y_no_ris.append(UAV_trajectory_no_ris[e,slot, 1])
                    z_no_ris.append(UAV_trajectory_no_ris[e,slot, 2])

            ax.scatter(w_k[:, 0], w_k[:, 1], c='k', marker='x',s = 40,label=u"")
            ax.scatter(0, 0, 50, c='g', marker='D', s = 60, label=u"")
            ax.plot(x_ris[:], y_ris[:], z_ris[:], c='g',linestyle='-', marker='', label=u"RIS-Assisted UAV")
            ax.plot(x_ris_no_shift[:], y_ris_no_shift[:], z_ris_no_shift[:], c='b', linestyle='-', marker='',label=u"UAV-R/P")
            ax.plot(x_no_ris[:], y_no_ris[:], z_no_ris[:], c='r',linestyle='-', marker='', label=u"UAV/R")
            ax.set_zlim(0, 250)
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000)

            font = {'family': 'Times New Roman',
                    'weight': 'normal',
                    'size': 12,
                    }

            ax.set_xlabel('X',font)
            ax.set_ylabel('Y',font)
            ax.set_zlabel('Z',font)
            plt.legend(prop=font,loc='upper right', shadow=True)
            formated_str = "UAV_GT_3D_%d.eps" %e
            plt.savefig(formated_str)
            plt.show()

            plt.plot(x_ris[:], y_ris[:],c='g',linestyle='-', marker='', label=u"RIS-Assisted UAV")
            plt.plot(x_ris_no_shift[:], y_ris_no_shift[:],c='b',linestyle='-', marker='',label=u"UAV-R/P")
            plt.plot(x_no_ris[:], y_no_ris[:],c='r',linestyle='-', marker='',label=u"UAV/R")
            plt.scatter(w_k[:, 0], w_k[:, 1], c='k',s = 40, marker='x',label=u"")
            plt.scatter(500, 500, c='g', marker='D', s = 60, label=u"")

            plt.ylabel(u'x(m)', font)
            plt.xlabel(u'y(m)', font)
            plt.legend(prop=font,loc='lower right', shadow=True)
            plt.grid()
            formated_str = "UAV_GT_2D_%d.eps" %e
            plt.savefig(formated_str)
            plt.show()
        return

    def plot_propulsion_energy(self,UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris,eps,slot_ris,slot_ris_no_shift,slot_no_ris):
        PEnergy_ris=self.env.flight_energy(UAV_trajectory_ris,UAV_flight_time_ris,eps,slot_ris)
        PEnergy_ris_no_shift = self.env.flight_energy(UAV_trajectory_ris_no_shift, UAV_flight_time_ris_no_shift,eps,slot_ris_no_shift)
        PEnergy_no_ris = self.env.flight_energy(UAV_trajectory_no_ris, UAV_flight_time_no_ris,eps,slot_no_ris)

        plot_energy = np.zeros((3,eps),dtype=np.float)
        for i in range(eps):
            plot_energy[0,i] = plot_energy[0,i]+np.sum(PEnergy_ris[i,:])/1000
            plot_energy[1, i] = plot_energy[1, i] + np.sum(PEnergy_ris_no_shift[i, :])/1000
            plot_energy[2, i] = plot_energy[2, i] + np.sum(PEnergy_no_ris[i, :])/1000

        myfont = matplotlib.font_manager.FontProperties(
            fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")

        res_1 = stats.relfreq(plot_energy[0, :], numbins=25)
        x_1 = res_1.lowerlimit + np.linspace(0, res_1.binsize * res_1.frequency.size, res_1.frequency.size)
        y_1 = np.cumsum(res_1.frequency)
        res_2 = stats.relfreq(plot_energy[1, :], numbins=25)
        x_2 = res_2.lowerlimit + np.linspace(0, res_2.binsize * res_2.frequency.size, res_2.frequency.size)
        y_2 = np.cumsum(res_2.frequency)
        res_3 = stats.relfreq(plot_energy[2, :], numbins=25)
        x_3 = res_3.lowerlimit + np.linspace(0, res_3.binsize * res_3.frequency.size, res_3.frequency.size)
        y_3 = np.cumsum(res_3.frequency)

        plt.plot(x_1, y_1, c='g', linestyle='-', marker='<', label=u"RIS-Assisted UAV")
        plt.plot(x_2, y_2, c='b', linestyle='-', marker='>', label=u"UAV-R/P")
        plt.plot(x_3, y_3, c='r', linestyle='-', marker='o', label=u"UAV/R")
        #plt.plot(np.arange(eps), plot_energy[0,:].T, c='r', linestyle='-', marker='<',label=u"RIS-Assisted UAV")
        #plt.plot(np.arange(eps), plot_energy[1,:].T, c='b', linestyle='-', marker='>',label=u"RIS-Assisted UAV without passive shift")
        #plt.plot(np.arange(eps), plot_energy[2,:].T, c='g', linestyle='-', marker='o',label=u"UAV system without RIS")

        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 12,
                }

        plt.xlabel(u'Propulsion Energy(KJ)', font)
        plt.ylabel(u'CDF', font)
        plt.legend(prop=font)
        plt.grid()
        plt.savefig('PE.eps')
        plt.show()

        sum_ris = np.sum(plot_energy[0,:])/eps
        sum_ris_no_shift = np.sum(plot_energy[1,:])/eps
        sum_no_ris = np.sum(plot_energy[2,:])/eps
        print("Propulsion Energy: RIS:%f;RIS_NO_SHIFT:%f;NO_RIS:%f" %(sum_ris, sum_ris_no_shift,sum_no_ris))
        return

    def plot_data_throughput(self, UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,GT_schedule_ris,GT_schedule_ris_no_shift,GT_schedule_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris,eps,slot_ris,slot_ris_no_shift,slot_no_ris):
        [Th_ris,rate_ris] = self.env.throughput(UAV_trajectory_ris, UAV_flight_time_ris,GT_schedule_ris,eps,1,1,slot_ris)
        [Th_ris_no_shift,rate_ris_no_shift] = self.env.throughput(UAV_trajectory_ris_no_shift, UAV_flight_time_ris_no_shift,GT_schedule_ris_no_shift,eps,1,0,slot_ris_no_shift)
        [Th_no_ris,rate_no_ris] = self.env.throughput(UAV_trajectory_no_ris, UAV_flight_time_no_ris,GT_schedule_no_ris,eps,0,0,slot_no_ris)

        plot_Th = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_Th[0, i] = plot_Th[0, i] + np.sum(Th_ris[i, :])
            plot_Th[1, i] = plot_Th[1, i] + np.sum(Th_ris_no_shift[i, :])
            plot_Th[2, i] = plot_Th[2, i] + np.sum(Th_no_ris[i, :])

        plot_Dr = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_Dr[0, i] = plot_Dr[0, i] + np.sum(rate_ris[i, :])
            plot_Dr[1, i] = plot_Dr[1, i] + np.sum(rate_ris_no_shift[i, :])
            plot_Dr[2, i] = plot_Dr[2, i] + np.sum(rate_no_ris[i, :])
        plot_Dr=plot_Dr/eps

        myfont = matplotlib.font_manager.FontProperties(
            fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")

        res_1 = stats.relfreq(plot_Th[0,:], numbins=25)
        x_1 = res_1.lowerlimit + np.linspace(0, res_1.binsize * res_1.frequency.size, res_1.frequency.size)
        y_1 = np.cumsum(res_1.frequency)
        res_2 = stats.relfreq(plot_Th[1,:], numbins=25)
        x_2 = res_2.lowerlimit + np.linspace(0, res_2.binsize * res_2.frequency.size, res_2.frequency.size)
        y_2 = np.cumsum(res_2.frequency)
        res_3 = stats.relfreq(plot_Th[2,:], numbins=25)
        x_3 = res_3.lowerlimit + np.linspace(0, res_3.binsize * res_3.frequency.size, res_3.frequency.size)
        y_3 = np.cumsum(res_3.frequency)

        #plt.plot(range(eps), plot_Th[0,:].T, c='r', linestyle='-', marker='<',label=u"RIS-Assisted UAV")
        #plt.plot(range(eps), plot_Th[1,:].T, c='b', linestyle='-', marker='>',label=u"RIS-Assisted UAV without passive shift")
        #plt.plot(range(eps), plot_Th[2,:].T, c='g', linestyle='-', marker='o',label=u"UAV system without RIS")
        plt.plot(x_1, y_1, c='g', linestyle='-', marker='<',label=u"RIS-Assisted UAV")
        plt.plot(x_2, y_2, c='b', linestyle='-', marker='>',label=u"UAV-R/P")
        plt.plot(x_3, y_3, c='r', linestyle='-', marker='o',label=u"UAV/R")

        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 12,
                }

        plt.xlabel(u'Throughput(Kbs)', font)
        plt.ylabel(u'CDF', font)
        plt.legend(prop=font)
        plt.grid()
        plt.savefig('Th.eps')
        plt.show()

        res_1 = stats.relfreq(plot_Dr[0, :], numbins=25)
        x_1 = res_1.lowerlimit + np.linspace(0, res_1.binsize * res_1.frequency.size, res_1.frequency.size)
        y_1 = np.cumsum(res_1.frequency)
        res_2 = stats.relfreq(plot_Dr[1, :], numbins=25)
        x_2 = res_2.lowerlimit + np.linspace(0, res_2.binsize * res_2.frequency.size, res_2.frequency.size)
        y_2 = np.cumsum(res_2.frequency)
        res_3 = stats.relfreq(plot_Dr[2, :], numbins=25)
        x_3 = res_3.lowerlimit + np.linspace(0, res_3.binsize * res_3.frequency.size, res_3.frequency.size)
        y_3 = np.cumsum(res_3.frequency)

        plt.plot(x_1, y_1, c='g', linestyle='-', marker='<', label=u"RIS-Assisted UAV")
        plt.plot(x_2, y_2, c='b', linestyle='-', marker='>', label=u"UAV-R/P")
        plt.plot(x_3, y_3, c='r', linestyle='-', marker='o', label=u"UAV/R")
        #plt.plot(range(eps), plot_Dr[0, :].T, c='g',linestyle='-', marker='<', label=u"RIS-Assisted UAV")
        #plt.plot(range(eps), plot_Dr[1, :].T, c='b', linestyle='-', marker='>',label=u"RIS-Assisted UAV without passive shift")
        #plt.plot(range(eps), plot_Dr[2, :].T, c='r', linestyle='-', marker='o',label=u"UAV system without RIS")
        plt.xlabel(u'Data Rate(kbps)', font)
        plt.ylabel(u'CDF', font)
        plt.legend(prop=font)
        plt.grid()
        plt.savefig('Dr.eps')
        plt.show()

        sum_ris = np.sum(plot_Th[0,:])/eps
        sum_ris_no_shift = np.sum(plot_Th[1,:])/eps
        sum_no_ris =np.sum(plot_Th[2,:])/eps
        print("Average Throughput: RIS:%f;RIS_NO_SHIFT:%f;NO_RIS:%f" %(sum_ris,sum_ris_no_shift,sum_no_ris))

        ave_ris = np.sum(plot_Dr[0,:])/eps
        ave_ris_no_shift = np.sum(plot_Dr[1,:])/eps
        ave_no_ris = np.sum(plot_Dr[2,:])/eps
        print("Average Data Rate: : RIS:%f;RIS_NO_SHIFT:%f;NO_RIS:%f" %(ave_ris,ave_ris_no_shift,ave_no_ris))
        return

    def plot_energy_efficiency(self, UAV_trajectory_ris,UAV_trajectory_ris_no_shift,UAV_trajectory_no_ris,GT_schedule_ris,GT_schedule_ris_no_shift,GT_schedule_no_ris,UAV_flight_time_ris,UAV_flight_time_ris_no_shift,UAV_flight_time_no_ris,eps,slot_ris,slot_ris_no_shift,slot_no_ris):
        [Th_ris,rate_ris] = self.env.throughput(UAV_trajectory_ris, UAV_flight_time_ris,GT_schedule_ris,eps,1,1,slot_ris)
        [Th_ris_no_shift,rate_ris_no_shift] = self.env.throughput(UAV_trajectory_ris_no_shift, UAV_flight_time_ris_no_shift,GT_schedule_ris_no_shift,eps,1,0,slot_ris_no_shift)
        [Th_no_ris,rate_no_ris] = self.env.throughput(UAV_trajectory_no_ris, UAV_flight_time_no_ris,GT_schedule_no_ris,eps,0,0,slot_no_ris)
        PEnergy_ris=self.env.flight_energy(UAV_trajectory_ris,UAV_flight_time_ris,eps,slot_ris)
        PEnergy_ris_shift = self.env.flight_energy(UAV_trajectory_ris_no_shift, UAV_flight_time_ris_no_shift,eps,slot_ris_no_shift)
        PEnergy_no_ris = self.env.flight_energy(UAV_trajectory_no_ris, UAV_flight_time_no_ris,eps,slot_no_ris)

        plot_ee = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_ee[0, i] = 1000*np.sum(Th_ris[i,:])/np.sum(PEnergy_ris[i, :])
            plot_ee[1, i] = 1000*np.sum(Th_ris_no_shift[i,:])/np.sum(PEnergy_ris_shift[i, :])
            plot_ee[2, i] = 1000*np.sum(Th_no_ris[i,:])/np.sum(PEnergy_no_ris[i, :])

        myfont = matplotlib.font_manager.FontProperties(
            fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
        res_1 = stats.relfreq(plot_ee[0, :], numbins=25)
        x_1 = res_1.lowerlimit + np.linspace(0, res_1.binsize * res_1.frequency.size, res_1.frequency.size)
        y_1 = np.cumsum(res_1.frequency)
        res_2 = stats.relfreq(plot_ee[1, :], numbins=25)
        x_2 = res_2.lowerlimit + np.linspace(0, res_2.binsize * res_2.frequency.size, res_2.frequency.size)
        y_2 = np.cumsum(res_2.frequency)
        res_3 = stats.relfreq(plot_ee[2, :], numbins=25)
        x_3 = res_3.lowerlimit + np.linspace(0, res_3.binsize * res_3.frequency.size, res_3.frequency.size)
        y_3 = np.cumsum(res_3.frequency)

        plt.plot(x_1, y_1, c='g', linestyle='-', marker='<', label=u"RIS-Assisted UAV")
        plt.plot(x_2, y_2, c='b', linestyle='-', marker='>', label=u"UAV-R/P")
        plt.plot(x_3, y_3, c='r', linestyle='-', marker='o', label=u"UAV/R")
        #plt.plot(range(eps), plot_ee[0,:].T, c='r',linestyle='-', marker='<', label=u"RIS-Assisted UAV")
        #plt.plot(range(eps), plot_ee[1,:].T, c='b', linestyle='-', marker='>',label=u"RIS-Assisted UAV without passive shift")
        #plt.plot(range(eps), plot_ee[2,:].T, c='g',linestyle='-', marker='o', label=u"UAV system without RIS")
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 12,
                }

        plt.xlabel(u'Energy-Efficiency(bits/J)', font)
        plt.ylabel(u'CDF', font)
        plt.legend(prop=font)
        plt.grid()
        plt.savefig('EE.eps')
        plt.show()

        ave_ris = np.sum(plot_ee[0, :]) / eps
        ave_ris_no_shift = np.sum(plot_ee[1, :]) / eps
        ave_no_ris = np.sum(plot_ee[2, :]) / eps
        print("Energy efficieny: : RIS:%f;RIS_NO_SHIFT:%f;NO_RIS:%f" % (ave_ris, ave_ris_no_shift, ave_no_ris))
        return



