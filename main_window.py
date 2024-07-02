import sys
import os

# Requires downloading TimeTagger software from Swabian (https://www.swabianinstruments.com/time-tagger/downloads/)
import TimeTagger
import time
import pyperclip
import threading
import pickle
from fig_utils import *
from threading import Thread

# Requires downloading Kinesis software from Thorlabs (https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=10285)
from pylablib.devices import Thorlabs
from PyQt5 import QtWidgets, uic
from LedIndicatorWidget import *
from pyqtconfig import ConfigManager
from datetime import date, datetime

import pyautogui

from widgets.CountsGraph import CountsGraph
from widgets.HOMGraph import HOMGraph
from widgets.Integrated_g2_Graph import Integrated_g2_Graph
from widgets.g2Graph import g2Graph
from widgets.utils import alphabetical_widget_list

matplotlib.use('Qt5Agg')

set_font_size(8)

config = ConfigManager()

ROUND = 5

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('ui/SpectralHOMGUI.ui', self, package='modules') # Load the .ui file

        """ File utility etc. """ # FFFFF
        self.selectRootPath_btn.clicked.connect(self.get_root_path)
        self.stageScan_selectRootPath_btn.clicked.connect(self.get_root_path_stage)
        self.homScan_selectRootPath_btn.clicked.connect(self.get_root_path_hom)

        self.tpx_scan_start.valueChanged.connect(self.update_tpx_scan_info)
        self.tpx_scan_end.valueChanged.connect(self.update_tpx_scan_info)
        self.tpx_scan_nsteps.valueChanged.connect(self.update_tpx_scan_info)
        self.tpx_scan_acq_time.valueChanged.connect(self.update_tpx_scan_info)
        self.tpx_scan_pause_time.valueChanged.connect(self.update_tpx_scan_info)
        self.tpx_scan_repeat_thresh.valueChanged.connect(self.update_tpx_scan_info)
        self.tpx_start_step_num.valueChanged.connect(self.update_tpx_scan_info)

        """ Connect Thorlabs devices """
        self.connectStage_btn.clicked.connect(self.connect_stage)
        self.devices = Thorlabs.list_kinesis_devices()

        """ Thorlabs MTS50-Z8 stage """ # SSSSS
        # delay_stage_index = 2
        # stage_name, stage_type = self.devices[delay_stage_index]
        # stage_name = '27601378' #[x for x, y in enumerate(self.devices) if y[1] == '27255970']
        self.stage_name = '27601378' #[x for x, y in enumerate(self.devices) if y[1] == '27255970']
        self.stage_scale = 'Z825'
        self.stage = None

        self.hom_scan_range.valueChanged.connect(self.set_hom_relative_range)
        self.ref_position_selection.buttonClicked.connect(self.update_reference_positions)
        self.ref_position_1.valueChanged.connect(self.update_reference_positions)
        self.ref_position_2.valueChanged.connect(self.update_reference_positions)
        self.ref_position_3.valueChanged.connect(self.update_reference_positions)
        self.velo_val.valueChanged.connect(self.set_velocity_params)
        self.accel_val.valueChanged.connect(self.set_velocity_params)

        # self.QuitButton.clicked.connect(QtCore.QCoreApplication.instance().quit)
        self.QuitButton.clicked.connect(self.quit)
        self.moveTo_btn.clicked.connect(self.move_to)
        self.moveTo_t0_btn.clicked.connect(self.move_to_t0)
        self.stageHome_btn.clicked.connect(self.stage_home)
        self.stageStop_btn.clicked.connect(self.stage_stop)
        self.set_t0_btn.clicked.connect(self.set_time_zero)

        self.moveBy_left_btn.clicked.connect(lambda: self.move_by(left_or_right=-1))
        self.moveBy_right_btn.clicked.connect(lambda: self.move_by(left_or_right=1))

        """ TimeTagger20 controls """ # TTTTT
        self.TT_error = ''
        self.connectTT_btn.clicked.connect(self.connect_tt)
        self.ttErrorText.setText(str(self.TT_error))
        self.active_channels = np.array([1, 2], dtype=np.int64) # just sets a default for the channels to be 1 and 2
        self.dpi = 150
        set_font_size(6)
        # self.chart_counts_frame.addWidget('a')
        self.chart_counts_stop_flag = threading.Event()
        self.chart_counts_graph = CountsGraph(parent=self, nchannels=8, width=self.chart_counts_frame.width() / self.dpi, height=self.chart_counts_frame.height() / self.dpi, dpi=self.dpi)
        self.chart_counts_graph.move(self.chart_counts_frame.pos())

        graph_size_scale = 0.8
        self.g2_graph = g2Graph(
            parent=self,
            nchannels=4,
            width=self.g2_tab.width()/self.dpi*graph_size_scale,
            height=self.g2_tab.height()/self.dpi*graph_size_scale,
            dpi=self.dpi
        )
        # self.g2_graph_toolbar = NavigationToolbar(self.chart_counts_graph, self)
        g2_layout = QVBoxLayout()
        g2_layout.addWidget(self.g2_graph)
        self.g2_tab.setLayout(g2_layout)
        # self.g2_graph.move(self.g2_frame.pos())

        self.hom_graph = HOMGraph(
            parent=self,
            nchannels=4,
            width=self.hom_tab.width()/self.dpi*graph_size_scale,
            height=self.hom_tab.height()/self.dpi*graph_size_scale,
            dpi=self.dpi
        )
        layout_hom = QVBoxLayout()
        layout_hom.addWidget(self.hom_graph)
        self.hom_tab.setLayout(layout_hom)

        self.intg2_graph = Integrated_g2_Graph(
            parent=self,
            width=self.hom_tab.width()/self.dpi*graph_size_scale,
            height=self.hom_tab.height()/self.dpi*graph_size_scale,
            dpi=self.dpi
        )
        layout_intg2 = QVBoxLayout()
        layout_intg2.addWidget(self.intg2_graph)
        self.integrated_g2_tab.setLayout(layout_intg2)

        self.testSignal_on_off_btn.clicked.connect(self.testSignal_on_off)
        self.chartCounts_btn.clicked.connect(self.chart_counts)
        self.correlate_btn.clicked.connect(lambda: self.correlate(self.correlate_btn.isChecked()))

        self.cps = alphabetical_widget_list(self.cps_box.children())
        self.trigger_levels = alphabetical_widget_list(self.trigger_box.children())
        self.channel_delays = alphabetical_widget_list(self.delay_box.children())
        self.deadtimes = alphabetical_widget_list(self.deadtime_box.children())
        self.channel_onoff = alphabetical_widget_list(self.ch_box.children())
        self.fileWriter_btn.clicked.connect(self.start_file_writer_thread)
        self.stage_scan_start_btn.clicked.connect(self.start_stage_scan_thread)
        self.hom_scan_start_btn.clicked.connect(self.start_hom_scan_thread)
        self.hom_scan_continuous_start_btn.clicked.connect(self.start_hom_scan_continuous_thread)
        self.tpx_scan_start_btn.clicked.connect(self.start_tpx_scan_thread)
        self.save_g2_btn.clicked.connect(self.save_g2)

        self.intg2start_btn.clicked.connect(lambda: self.start_int_g2_thread(self.intg2start_btn.isChecked()))
        ### HANDLERS FOR DEFAULT VALUES ###
        """ 
        The config file is saved everytime the program is closed, storing the last recorded values of whatever variables
        it was given a handler for. Just need to provide the object attribute you want to record (e.g. self.stage_position)
        and it will automatically load that value next time the program is opened
        """

        # Misc
        config.add_handler('root_path', self.rootpath)
        config.add_handler('filename', self.filename)
        config.add_handler('notes', self.notes)

        # Stage
        config.add_handler('stage_scan_filename', self.stage_scan_filename)
        config.add_handler('stage_scan_rootpath', self.stage_scan_rootpath)
        config.add_handler('stage_scan_start', self.stage_scan_start)
        config.add_handler('stage_scan_end', self.stage_scan_end)
        config.add_handler('stage_scan_nsteps', self.stage_scan_nsteps)
        config.add_handler('stage_scan_acqt', self.stage_scan_acqt)
        config.add_handler('stage_scan_waitt', self.stage_scan_waitt)
        config.add_handler('stage_scan_waitt', self.stage_scan_waitt)
        config.add_handler('stage_time_zero', self.stage_time_zero)
        config.add_handler('ref_position_1', self.ref_position_1)
        config.add_handler('ref_position_2', self.ref_position_2)
        config.add_handler('ref_position_3', self.ref_position_3)
        config.add_handler('velo_val', self.velo_val)
        config.add_handler('accel_val', self.accel_val)

        # Chart counts
        config.add_handler('counter_acquisition_t', self.counter_acquisition_t)
        config.add_handler('counter_tmax', self.counter_tmax)
        config.add_handler('chart_coinc', self.chartCoincidences)

        # g2
        config.add_handler('g2_binwidth', self.g2_binwidth)
        config.add_handler('g2_nbins', self.g2_nbins)
        config.add_handler('time_zero', self.stage_time_zero)
        config.add_handler('intg2_acqt', self.intg2_acqt)
        config.add_handler('intg2_npoints', self.intg2_npoints)
        config.add_handler("coinc_window", self.coinc_window)

        # HOM
        config.add_handler('hom_scan_filename', self.hom_scan_filename)
        config.add_handler('hom_scan_rootpath', self.hom_scan_rootpath)
        config.add_handler('hom_scan_start', self.hom_scan_start)
        config.add_handler('hom_scan_end', self.hom_scan_end)
        config.add_handler('hom_scan_nsteps', self.hom_scan_nsteps)
        config.add_handler('hom_scan_acqt', self.hom_scan_acqt)
        config.add_handler('hom_scan_range', self.hom_scan_range)
        config.add_handler('hom_scan_start_rel', self.hom_scan_start_rel)
        config.add_handler('hom_scan_end_rel', self.hom_scan_end_rel)
        config.add_handler('hom_scan_pause_t', self.hom_scan_pause_t)

        # TimePix
        config.add_handler('tpx_scan_start', self.tpx_scan_start)
        config.add_handler('tpx_scan_end', self.tpx_scan_end)
        config.add_handler('tpx_scan_nsteps', self.tpx_scan_nsteps)
        config.add_handler('tpx_scan_acq_time', self.tpx_scan_acq_time)
        config.add_handler('tpx_scan_pause_time', self.tpx_scan_pause_time)
        config.add_handler('tpx_scan_repeat_thresh', self.tpx_scan_repeat_thresh)
        config.add_handler('sophy_start_x', self.sophy_start_x)
        config.add_handler('sophy_start_y', self.sophy_start_y)

        # Looping through the self.trigger_levels GroupBox object, instead of assigning handlers individually
        for i, ch in enumerate(self.trigger_levels):
            config.add_handler('ch{}_trig'.format(i), ch)

        for i, ch in enumerate(self.channel_delays):
            config.add_handler('ch{}_delay'.format(i), ch)

        for i, ch in enumerate(self.deadtimes):
            config.add_handler('ch{}_deadtime'.format(i), ch)

        for i, ch in enumerate(self.channel_onoff):
            config.add_handler('ch{}_checkBox'.format(i), ch)

        # for i, ch in enumerate(self.g2_matrix):
        #     config.add_handler('matrix{}_entry'.format(i), ch)

        self.show() # Show the GUI
        self.testButton.clicked.connect(self.test)
        self.testButton2.clicked.connect(self.update_channel_settings)
        self.loadDefaults_btn.clicked.connect(self.load_default_variables)
        self.load_default_variables()

    """ File utility functions """ #fFFFFF
    def load_default_variables(self):
        with open('config/SpectralHOMGUI_config', 'rb') as f:
            self.config_defaults = pickle.load(f)
        config.set_defaults(self.config_defaults)
        for key, value in self.config_defaults.items():
            config.set(key, value)

    def save_variables(self):
        config_dict = config.as_dict()
        with open('config/SpectralHOMGUI_config', 'wb') as f:
            pickle.dump(config_dict, f)

    def get_root_path(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.rootpath.setPlainText(folderpath)

    def get_root_path_stage(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.stage_scan_rootpath.setPlainText(folderpath)

    def get_root_path_hom(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.hom_scan_rootpath.setPlainText(folderpath)

    def get_filepath(self):
        """ A helper function that grabs the text from the rootpath and filename boxes, so they don't need to be
        accessesd everytime they're used """
        root = self.rootpath.toPlainText()
        fname = self.filename.toPlainText()
        path = os.path.join(root, fname)
        return path

    def get_filepath_stage_scan(self):
        root = self.stage_scan_rootpath.toPlainText()
        fname = self.stage_scan_filename.toPlainText()
        path = os.path.join(root, fname)
        return path

    def get_filepath_hom_scan(self):
        root = self.hom_scan_rootpath.toPlainText()
        fname = self.hom_scan_filename.toPlainText()
        path = os.path.join(root, fname)
        return path

    def test(self):
        # print('testing')
        # print(self.get_filepath())
        config.set('text', 'new value')

    def quit(self):
        try:
            TimeTagger.freeTimeTagger(self.tagger)
        except Exception as e:
            print('Error disconnecting Time Tagger')
        self.save_variables()
        self.close()

    """ Stage functions """ #fSSSSS
    def connect_stage(self):
        try:
            self.stage = Thorlabs.KinesisMotor(
                self.stage_name,
                self.stage_scale
            )

            self.stageConnect_led.setChecked(self.stage.is_opened())
            self.scale = 1000
            self.stage_error = ''

            self.start_time = time.time()
            self.pos_thread = Thread(target=self.update_stage_pos)
            self.pos_thread.daemon = True
            self.pos_thread.start()

            self.update_velo_params = False
            self.queue_stage_stop = False
            self.queue_stage_home = False

        except Exception as e:
            # Thorlabs.KinesisMotor(self.stage_name, self.stage_scale).close()
            # self.stage = Thorlabs.KinesisMotor(self.stage_name, self.stage_scale)
            print(e)


    def set_velocity_params(self):
        if self.stage is not None:
            if not self.is_moving:
                try:
                    # self.velocity = float(self.velo_val.text())/self.scale
                    # self.acceleration = float(self.accel_val.text())/self.scale
                    self.velocity = self.velo_val.value()/self.scale
                    self.acceleration = self.accel_val.value()/self.scale
                    self.update_velo_params = True
                    # self.s.setup_velocity(max_velocity=self.velocity, acceleration=self.acceleration)
                    self.stage_error = ''
                except Exception as e:
                    self.stage_error = e
                    print(e)

    def move_to(self):
        val = float(self.moveTo_val.text())/self.scale
        self.stage.move_to(val)

    def move_to_t0(self):
        self.stage.move_to(self.stage_time_zero.value() / self.scale)

    def move_by(self, left_or_right):
        val = float(self.moveBy_val.text())/self.scale*left_or_right
        self.stage.move_by(val)

    def set_time_zero(self):
        self.stage_time_zero.setValue(self.current_stage_position)

    def update_reference_positions(self):
        if self.ref_position_selection.checkedButton().text() == '1':
            self.stage_time_zero.setValue(self.ref_position_1.value())
        elif self.ref_position_selection.checkedButton().text() == '2':
            self.stage_time_zero.setValue(self.ref_position_2.value())
        elif self.ref_position_selection.checkedButton().text() == '3':
            self.stage_time_zero.setValue(self.ref_position_3.value())

    def update_tpx_scan_info(self):
        start_step = int(self.tpx_start_step_num.value())

        step_size = (self.tpx_scan_end.value() - self.tpx_scan_start.value()) / self.tpx_scan_nsteps.value()
        est_duration = (self.tpx_scan_acq_time.value() + self.tpx_scan_pause_time.value()) * (self.tpx_scan_nsteps.value() - start_step) / 60
        self.tpx_step_size.setText(str(step_size.__round__(ROUND)))
        self.tpx_scan_duration.setText(str(est_duration.__round__(1)))

        acq_time = self.tpx_scan_acq_time.value()
        repeat_thresh = self.tpx_scan_repeat_thresh.value()
        if acq_time <= repeat_thresh:
            repeat = 0
        elif acq_time > repeat_thresh:
            repeat = (acq_time//repeat_thresh)
            acq_time = repeat_thresh

        self.tpx_scan_params = {
            'stage_start': self.tpx_scan_start.value(),
            'stage_end': self.tpx_scan_end.value(),
            'nsteps': int(self.tpx_scan_nsteps.value()),
            'exposure_time': acq_time,
            'repeat_threshold': self.tpx_scan_repeat_thresh.value(),
            'repeat': repeat,
            'step_size': step_size,
            'estimated_duration': est_duration,
            'start_step': start_step
        }

        self.tpx_scan_nrepeats.setText(str(int(repeat)))
        self.tpx_scan_acq_time_total.setText(str(int(acq_time*repeat)))

    def stage_home(self):
        self.queue_stage_home = True

    def stage_stop(self):
        self.queue_stage_stop = True

    def update_stage_pos(self):
        while True:
            try:
                pos = round(self.stage.get_position() * self.scale, ROUND)
                self.current_stage_position = pos
                self.stagePosition.setText(str(pos))
                self.is_moving = self.stage.is_moving()
                self.stageMoving_led.setChecked(self.is_moving)
                self.stageErrorText.setText(str(self.stage_error))

                if self.update_velo_params:
                    self.stage.setup_velocity(max_velocity=self.velocity, acceleration=self.acceleration)
                    print('Velocity parameters updated')
                    self.update_velo_params = False

                if self.queue_stage_stop:
                    self.stage.stop()
                    print('Stage motion stopped')
                    self.queue_stage_stop = False

                if self.queue_stage_home:
                    self.stage.home()
                    self.stage.wait_for_home()
                    print('Stage homed')
                    self.queue_stage_home = False

                interval = float(0.10)
                time.sleep(float(interval))
            except Exception as e:
                self.stage_error = e
                print(e)


    """ TimeTagger functions """ #fTTTTT
    def connect_tt(self):
        self.tagger = TimeTagger.createTimeTagger()
        self.ttConnect_led.setChecked(True)
        self.update_channel_settings()

    def reset_tt(self):
        TimeTagger.freeTimeTagger(self.tagger)
        self.ttConnect_led.setChecked(False)
        time.sleep(0.1)
        self.connect_tt()

    def disconnect_tt(self):
        TimeTagger.freeTimeTagger(self.tagger)
        del self.tagger
        self.ttConnect_led.setChecked(False)

    def testSignal_on_off(self):
        on_or_off = self.testSignal_on_off_btn.isChecked()
        for ch in self.active_channels:
            self.tagger.setTestSignal(ch, on_or_off)

    def get_active_channels(self):
        # Extra 'False' is added to increase index by 1, since the TT labels channels 1 - 8
        channel_list = [False, self.ch1_checkBox.isChecked(), self.ch2_checkBox.isChecked(), self.ch3_checkBox.isChecked(), self.ch4_checkBox.isChecked(),
                               self.ch5_checkBox.isChecked(), self.ch6_checkBox.isChecked(), self.ch7_checkBox.isChecked(), self.ch8_checkBox.isChecked()
                        ]
        self.active_channels = np.where(channel_list)[0]

    def update_channel_settings(self):
        for i in range(0, 8):
            self.tagger.setTriggerLevel(channel=i+1, voltage=self.trigger_levels[i].value())
            self.tagger.setInputDelay(channel=i+1, delay=self.channel_delays[i].value())
            self.tagger.setDeadtime(channel=i+1, deadtime=self.deadtimes[i].value())

        self.tagger.setConditionalFilter(
            filtered=[int(self.trigger_ch.value())],
            trigger=self.active_channels[:-1],
        )
        print('Channel settings updated')

    def chart_counts(self):
        if self.chartCounts_btn.isChecked():
            self.chart_counts_stop_flag = threading.Event()
            self.chartCounts_led.setChecked(True)
            # Create a thread that will run the plot_chart_counts function
            thread = threading.Thread(target=self.plot_chart_counts)
            # Start the thread
            thread.start()

        elif not self.chartCounts_btn.isChecked():
            self.chart_counts_stop_flag.set()
            self.chartCounts_led.setChecked(False)


    # Function to stop the thread
    def stop_chart_counts_thread(self):
        # Set the stop flag to signal the thread to stop
        self.chart_counts_stop_flag.set()

    def plot_chart_counts(self):
        print('Thread started')

        chart_coincidences = self.chartCoincidences.isChecked()
        self.get_active_channels()
        ctr_acquisition_t = float(self.counter_acquisition_t.text())
        ctr_tmax = float(self.counter_tmax.text())
        self.ctr_binwidth = ctr_acquisition_t * 1e12
        self.ctr_nbins = ctr_tmax / ctr_acquisition_t

        # singles_channels = self.active_channels
        # singles_channels = [1, 2]
        singles_channels = self.active_channels
        # groups = list(itertools.combinations(singles_channels, 2))

        if chart_coincidences:
            coincidences_vchannels = TimeTagger.Coincidence(self.tagger, self.active_channels, coincidenceWindow=self.coinc_window.value())
            channels = [*singles_channels, *coincidences_vchannels.getChannels()]
        else:
            channels = [*singles_channels]

        self.all_channels = channels

        self.counter = TimeTagger.Counter(
            tagger=self.tagger,
            channels=self.all_channels,
            binwidth=self.ctr_binwidth,
            n_values=self.ctr_nbins
        )

        self.counter.start()

        interval = float(self.counter_acquisition_t.text())

        ctr = 0
        while True:
            # print(ctr)
            ctr += 1
            # Wait for acquistion time to accumulate counts
            time.sleep(float(interval))

            # Send data to graph to plot
            t = self.counter.getIndex()/1e12
            data = self.counter.getData()/interval

            plot_data = data.copy()
            plot_data[2, :] *= self.coinc_gain.value()
            # data[2, :] *= self.coinc_gain.value()

            self.chart_counts_graph.plot(
                t,
                plot_data,
                self.all_channels,
            )

            # Update the cps indicators
            self.update_cps(data)

            if self.chart_counts_stop_flag.is_set():  # Check if the stop flag is set
                print('Thread terminated.')
                return


    def update_cps(self, data):

        counter_average = int(self.counterAveraging.value())

        for i, ch in enumerate(self.active_channels):
            cps = data[i, -counter_average:-1].mean()
            self.cps[ch-1].setValue(cps)

        if len(self.active_channels) == 2:
            accidentals_channel_1 = self.cps[self.active_channels[0] - 1].value()
            accidentals_channel_2 = self.cps[self.active_channels[1] - 1].value()
            # self.accidentals.setValue((self.cps[0].value()*self.cps[1].value())*self.g2_binwidth.value()*self.g2_nbins.value()*1e-12)
            # self.accidentals.setValue(accidentals_channel_1*accidentals_channel_2*self.g2_binwidth.value()*self.g2_nbins.value()*1e-12)
            self.accidentals.setValue(accidentals_channel_1*accidentals_channel_2*2*self.coinc_window.value()*1e-12)
        else:
            self.accidentals.setValue(42)

        if self.chartCoincidences.isChecked():
            self.update_coincidences(data, counter_average)

    def update_coincidences(self, data, averaging=5):
        """ Currently this will only work when 2 channels are being used, as it assumes the data array consists of
        singles 1, singles 2, and coincidences 1/2 """
        singles1 = data[0, -averaging:-1].mean()
        singles2 = data[1, -averaging:-1].mean()
        coincs = data[2, -averaging:-1].mean()

        mean_square_singles = np.sqrt(singles1 * singles2)

        self.coincidences.setValue(coincs)

        if self.subtract_accidentals.isChecked():
            coincs = coincs - self.accidentals.value()
        if self.divide_by_singles.isChecked():
            # data = data / (np.sqrt(self.cps[0].value() * self.cps[1].value()))
            coincs = 100 * coincs / mean_square_singles

        self.coincidences_corr.setValue(coincs)

        if self.divide_by_singles.isChecked():
            self.coin_to_sing.setValue(coincs)
        else:
            self.coin_to_sing.setValue(100 * coincs / mean_square_singles)


    def get_g2_channel_pairs(self):
        self.start_ch = []
        self.stop_ch = []
        for i, ch_pair in enumerate(self.g2_matrix.children()):
            try:
                if ch_pair.isChecked():
                    pair = ch_pair.objectName().split('g2_')[-1].split('and')
                    self.start_ch.append(int(pair[0]))
                    self.stop_ch.append(int(pair[1]))
            except Exception as e:
                # This is to skip the label boxes in the correlation matrix group box
                print(e)
                pass


    def correlate(self, start, display_text=True, plot=True):
        # if self.correlate_btn.isChecked():
        if start:
            self.get_active_channels()
            binwidth = self.g2_binwidth.value()
            n_bins = self.g2_nbins.value()
            self.get_g2_channel_pairs()
            self.g2 = []
            for i in range(0, len(self.start_ch)):
                self.g2.append(TimeTagger.Correlation(self.tagger, channel_1=self.start_ch[i], channel_2=self.stop_ch[i], binwidth=binwidth, n_bins=n_bins))

            if plot:
                self.stop_g2_thread = False
                self.g2_thread = Thread(target=self.plot_g2)
                # self.counter_thread.daemon = True
                self.g2_thread.start()
                self.g2_led.setChecked(self.g2[0].isRunning())
            if display_text:
                print('Correlation started')

        # elif not self.correlate_btn.isChecked():
        if not start:
            self.g2_data = []
            self.g2_idxs = []
            for g2 in self.g2:
                g2.stop()
                self.g2_data.append(g2.getData())
                self.g2_idxs.append(g2.getIndex())
            self.g2_led.setChecked(self.g2[0].isRunning())
            self.stop_g2_thread = True
            del self.g2
            # self.reset_tt()
            if display_text:
                print('Correlation stopped')
        # self.stageMoving.setText(str(self.is_moving))

    def plot_g2(self):
        # print('Correlation thread started..')
        while True:
            try:
                if hasattr(self, 'g2'):
                    # Wait for acquistion time to accumulate counts
                    interval = float(self.counter_acquisition_t.text())
                    time.sleep(float(interval))
                    # Send data to graph to plot
                    t = []
                    data = []
                    for g2 in self.g2:
                        t.append(g2.getIndex())
                        data.append(g2.getData())

                    t = np.array(t[0]) # this assumes t is the same for all g2s, which should almost always be the case
                    data = np.array(data)
                    # t = self.g2.getIndex()/1e12
                    # data = self.g2.getData()
                    if data.ndim < 2:
                        data = np.expand_dims(data, 0)
                    self.g2_graph.plot(t, data)
                else:
                    break
            except Exception as e:
                self.TT_error = e
                self.ttErrorText.setText(str(self.TT_error))
                print(e)
                break

        if self.stop_g2_thread:
            print('g2 thread terminated.')
            return

    def start_file_writer_thread(self):
        self.file_writer_thread = Thread(target=self.file_writer)
        self.file_writer_thread.start()

    def file_writer(self):
        fname = self.get_filepath()
        tic = time.time()
        max_t = self.fileWriter_acquisitionTime.value()
        t_elapsed = 0
        self.filewriter = TimeTagger.FileWriter(self.tagger, fname, self.active_channels)
        self.fileWriter_led.setChecked(self.filewriter.isRunning())
        print('FileWriter started')

        while self.fileWriter_btn.isChecked() and t_elapsed < max_t:
            time.sleep(0.1)
            t_elapsed = time.time() - tic

        self.filewriter.stop()
        self.fileWriter_led.setChecked(self.filewriter.isRunning())
        self.fileWriter_btn.setChecked(False)
        print('FileWriter ended')

        return

    def save_g2(self):
        root = 'G:\Shared drives\JCEP Lab\Projects\Spectral HOM\g2s'
        fname = self.g2_filename.toPlainText()
        save_path = os.path.join(root, fname)
        g2_data = np.array(self.g2_data)
        g2_idxs = np.array(self.g2_idxs)

        # column_names = self.active_channels.astype(str) # TODO: make column names the g2 channel pairs
        df = pd.DataFrame(
            data=g2_data.T,
            index=g2_idxs[0],
            # columns=column_names
        )
        df.index.name = 'Time (ps)'
        df.to_csv(save_path)

    def start_stage_scan_thread(self):
        self.stage_scan_thread = Thread(target=self.stage_scan)
        self.stage_scan_thread.start()

    def stage_scan(self):
        start = self.stage_scan_start.value()
        end   = self.stage_scan_end.value()
        nsteps = int(self.stage_scan_nsteps.value())
        stage_steps = np.linspace(start, end, nsteps)
        cps_arr = np.zeros((len(self.active_channels), nsteps))
        pause = self.stage_scan_waitt.value()
        self.stage_scan_led.setChecked(True)

        for i, step in enumerate(stage_steps):
            if self.stage_scan_stop_btn.isChecked():
                self.stage_scan_stop_btn.setChecked(False)
                break

            self.stage.move_to(step / self.scale)
            self.is_moving = True
            ctr = 0
            while self.is_moving:
                ctr += 1
                time.sleep(0.01)

            time.sleep(pause)
            for j, ch in enumerate(self.active_channels):
                cps_arr[j, i] = self.cps[ch-1].value()

            if self.stage_scan_write_scan_checkBox.isChecked() and i % 10 == 0:
                print('Scan saved')
                column_names = self.active_channels.astype(str)
                df = pd.DataFrame(data=cps_arr.T,
                                  index=stage_steps,
                                  columns=column_names)
                df.index.name = 'Steps (mm)'
                save_path = self.get_filepath_stage_scan()
                df.to_csv(save_path)

        self.stage_scan_led.setChecked(False)
        self.stage_scan_start_btn.setChecked(False)
        return


    def set_hom_relative_range(self):
        start = self.stage_time_zero.value() - self.hom_scan_range.value()/2
        end = self.stage_time_zero.value() + self.hom_scan_range.value()/2
        self.hom_scan_start_rel.setText(
            str(start.__round__(ROUND))
        )
        self.hom_scan_end_rel.setText(
            str(end.__round__(ROUND))
        )

    def start_hom_scan_thread(self):
        self.hom_scan_thread = Thread(target=self.hom_scan)
        self.hom_scan_thread.start()

    def hom_scan(self):

        if self.hom_range_type.checkedButton().text() == 'Relative':
            start = float(self.hom_scan_start_rel.text())
            end   = float(self.hom_scan_end_rel.text())
        elif self.hom_range_type.checkedButton().text() == 'Absolute':
            start = self.hom_scan_start.value()
            end   = self.hom_scan_end.value()

        nsteps = int(self.hom_scan_nsteps.value())
        acqt = self.hom_scan_acqt.value()
        pause_t = self.hom_scan_pause_t.value()
        hom_steps = np.linspace(start, end, nsteps).round(ROUND)
        self.hom_scan_led.setChecked(True)

        self.velo_val.setValue(1)
        self.accel_val.setValue(1)
        time.sleep(0.1)
        self.stage.move_to(hom_steps[0] / self.scale)

        ctr = 0
        while self.is_moving:
            ctr += 1
            time.sleep(0.01)

        self.velo_val.setValue(self.hom_velo_val.value())
        self.accel_val.setValue(self.hom_accel_val.value())
        time.sleep(0.1)

        positions = []
        g2_zeros  = []
        for i, step in enumerate(hom_steps):
            if self.hom_scan_stop_btn.isChecked():
                self.hom_scan_stop_btn.setChecked(False)
                break

            self.stage.move_to(step / self.scale)
            self.is_moving = True
            ctr = 0
            while self.is_moving:
                ctr += 1
                time.sleep(0.01)

            # Pause to let stage stabilize
            time.sleep(pause_t)

            # Pause to acquire
            time.sleep(acqt)

            # Get data from g2
            # DDDDD
            data = self.coincidences.value()
            # data = np.sum(self.g2[0].getData()) / acqt # coincidences per second

            if self.subtract_accidentals.isChecked():
                data = data - self.accidentals.value()
            if self.divide_by_singles.isChecked():
                data = data / (np.sqrt(self.cps[self.active_channels[0] - 1].value() * self.cps[self.active_channels[1] - 1].value()))

            # FFFFF
            g2_zeros.append(data)

            positions.append(self.current_stage_position)

            if i > 0:
                self.hom_graph.plot(positions, g2_zeros)

            if self.hom_scan_write_scan_checkBox.isChecked() and i % 10 == 0:
                print('Scan saved')
                # column_names = self.active_channels.astype(str)
                df = pd.DataFrame(
                    data=g2_zeros,
                    index=positions
                )
                df.index.name = 'Steps (mm)'
                save_path = self.get_filepath_hom_scan()
                df.to_csv(save_path)

        self.hom_scan_led.setChecked(False)
        self.hom_scan_start_btn.setChecked(False)

        return

        # if self.fileWriter_btn.isChecked():
        #     self.filewriter = TimeTagger.FileWriter(self.tagger, fname, self.active_channels)
        #     print('FileWriter started')
        #     self.fileWriter_led.setChecked(self.filewriter.isRunning())
        # elif not self.fileWriter_btn.isChecked():
        #     self.filewriter.stop()
        #     print('FileWriter ended')
        #     self.fileWriter_led.setChecked(self.filewriter.isRunning())


    def start_hom_scan_continuous_thread(self):
        self.hom_scan_continuous_thread = Thread(target=self.hom_scan_continuous)
        self.hom_scan_continuous_thread.start()

    def hom_scan_continuous(self):
        if self.hom_range_type.checkedButton().text() == 'Relative':
            start = float(self.hom_scan_start_rel.text())
            end   = float(self.hom_scan_end_rel.text())
        elif self.hom_range_type.checkedButton().text() == 'Absolute':
            start = self.hom_scan_start.value()
            end   = self.hom_scan_end.value()

        self.hom_scan_led.setChecked(True)
        self.velo_val.setValue(1)
        self.accel_val.setValue(1)
        while self.update_velo_params == True:
            print('Waiting for velo params to update..')
            time.sleep(0.1)
        print('Velocity set to 1 for moving to starting position')

        time.sleep(0.1)
        self.stage.move_to(start/self.scale)

        # Wait to move to starting position
        print('Waiting for starting position..') #, end="\r")
        while self.current_stage_position.__round__(2) != start.__round__(2):
            time.sleep(0.1)

        print('Arrived at starting position')

        self.velo_val.setValue(self.hom_velo_val.value())
        self.accel_val.setValue(self.hom_accel_val.value())
        while self.update_velo_params == True:
            print('Waiting for velo params to update..')
            time.sleep(0.1)

        #GGGGG

        print('Velocity set for HOM scan')
        time.sleep(1)

        positions = []
        coincs = []


            # print(self.current_stage_position)
            # print('Waiting for starting position..') #, end="\r")

        print('Starting movement to end position')
        self.stage.move_to(end/self.scale)

        i = 0

        while self.current_stage_position.__round__(2) != end.__round__(2):
            if self.hom_scan_stop_btn.isChecked():
                self.hom_scan_stop_btn.setChecked(False)
                break

            # Pause to let stage stabilize
            time.sleep(float(self.counter_acquisition_t.text()))

            # Get data from g2
            data = self.coincidences.value()
            # data = np.sum(self.g2[0].getData()) / acqt # coincidences per second

            if self.subtract_accidentals.isChecked():
                data = data - self.accidentals.value()
            if self.divide_by_singles.isChecked():
                data = data / (np.sqrt(self.cps[self.active_channels[0] - 1].value() * self.cps[self.active_channels[1] - 1].value()))

            coincs.append(data)
            positions.append(self.current_stage_position)

            if i > 0:
                self.hom_graph.plot(positions, coincs)

            if self.hom_scan_write_scan_checkBox.isChecked() and i % 10 == 0:
                # print('Scan saved')
                # column_names = self.active_channels.astype(str)
                df = pd.DataFrame(
                    data=coincs,
                    index=positions
                )
                df.index.name = 'Steps (mm)'
                save_path = self.get_filepath_hom_scan()
                df.to_csv(save_path)

            i += 1

        self.hom_scan_led.setChecked(False)
        self.hom_scan_continuous_start_btn.setChecked(False)

        return

    def start_tpx_scan_thread(self):
        self.tpx_scan_thread = Thread(target=self.tpx_scan)
        self.tpx_scan_thread.start()

    def tpx_scan(self):
        # acq_time = self.tpx_scan_acq_time.value()
        # repeat_thresh = self.tpx_scan_repeat_thresh.value()
        # if acq_time <= repeat_thresh:
        #     repeat = 0
        # if acq_time > repeat_thresh:
        #     repeat = (acq_time//repeat_thresh)
        #     acq_time = repeat_thresh
        # self.tpx_scan_repeats = repeat
        # self.tpx_scan_acq_time = acq_time
        # self.tpx_scan_nrepeats.setText(str(int(repeat)))
        # self.tpx_scan_acq_time_total.settext(str(int(acq_time*repeat)))
        self.stop_tpx_scan_thread = False

        while True:
            acq_time = self.tpx_scan_params['exposure_time']
            repeat = self.tpx_scan_params['repeat']
            pause_time = self.tpx_scan_pause_time.value()
            click_x = self.sophy_start_x.value()
            click_y = self.sophy_start_y.value()

            if self.use_hom_scan_stage_positions.isChecked():
                if self.hom_range_type.checkedButton().text() == 'Relative':
                    start = float(self.hom_scan_start_rel.text())
                    end = float(self.hom_scan_end_rel.text())
                elif self.hom_range_type.checkedButton().text() == 'Absolute':
                    start = self.hom_scan_start.value()
                    end = self.hom_scan_end.value()
                nsteps = int(self.hom_scan_nsteps.value())
            else:
                start = self.tpx_scan_params['stage_start']
                end = self.tpx_scan_params['stage_end']
                nsteps = int(self.tpx_scan_nsteps.value())



            # nsteps = int(self.tpx_scan_nsteps.value())

            tpx_steps = np.linspace(
                start,
                end,
                nsteps
            ).round(ROUND)

            # skip start_step steps
            tpx_steps = tpx_steps[self.tpx_scan_params['start_step']:]

            self.tpx_scan_led.setChecked(True)

            self.velo_val.setValue(1)
            self.accel_val.setValue(1)
            time.sleep(0.1)
            self.stage.move_to(tpx_steps[0] / self.scale)

            # get info for header file
            pyautogui.click(1356, 1012)
            pyautogui.click(1630, 210)
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.hotkey('ctrl', 'c')
            root = pyperclip.paste()
            pyautogui.click(1630, 236)
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.hotkey('ctrl', 'c')
            fname = pyperclip.paste()
            pyautogui.click(1320, 1012)

            print(f"File: {fname}")

            # write header
            with open(f"{root}/{fname[:-1]}.config.txt", "w") as f:
                f.write(f"Date: {date.today()}\n")
                f.write(f"Start time: {datetime.now()}\n")
                f.write(f"Stage start (mm): {start}\n")
                f.write(f"Stage stop (mm): {end}\n")
                f.write(f"Num steps: {nsteps}\n")
                f.write(f"Step size: {self.tpx_step_size.text()}\n")
                f.write(f"Total acquisition time (s): {self.tpx_scan_acq_time.text()}\n")
                f.write(f"Individual exposure time (s): {acq_time}\n")
                f.write(f"Exposures per step: {max(repeat, 1)}\n")

            ctr = 0
            while self.is_moving:
                ctr += 1
                time.sleep(0.01)

            self.velo_val.setValue(self.hom_velo_val.value())
            self.accel_val.setValue(self.hom_accel_val.value())
            time.sleep(0.1)

            self.tpx_scan_progressBar.setRange(0, nsteps)
            for i, step in enumerate(tpx_steps):
                if self.stop_tpx_scan_thread:
                    print('Timepix scan thread terminated.')
                    return
                if self.tpx_scan_stop_btn.isChecked():
                    self.tpx_scan_stop_btn.setChecked(False)
                    self.stop_tpx_scan_thread = True
                    break

                self.stage.move_to(step / self.scale)
                self.is_moving = True

                ctr = 0
                while self.is_moving:
                    ctr += 1
                    time.sleep(0.01)

                # Pause to let stage stabilize
                time.sleep(pause_time)

                # Click start on Sophy
                if i == 0:
                    # Click to switch to Sophy window
                    pyautogui.doubleClick(
                        x=self.sophy_exp_time_x.value(),
                        y=self.sophy_exp_time_y.value()
                    )
                    pyautogui.typewrite(str(acq_time))
                    if repeat > 0:
                        pyautogui.click(
                            x=1320,
                            y=244
                        )
                        pyautogui.doubleClick(
                            x=1760,
                            y=248
                        )
                        pyautogui.typewrite(str(repeat))

                pyautogui.click(x=click_x, y=click_y)

                # Pause to acquire; 0.5s extra per exposure to account for Timepix measurement delays
                if repeat > 0:
                    time.sleep((acq_time + 0.5) * repeat)
                else:
                    time.sleep(acq_time + 0.5)

                self.tpx_scan_progressBar.setValue(i)
                print(f"loop {i} succesful")

            self.tpx_scan_progressBar.setValue(nsteps)

            if repeat > 0:
                pyautogui.click(
                    x=1320,
                    y=244
                )

            self.tpx_scan_led.setChecked(False)
            self.tpx_scan_start_btn.setChecked(False)
            self.stop_tpx_scan_thread = True

            return

    def start_int_g2_thread(self, start):
        if start:
            self.stop_intg2_thread = False
            self.integrated_g2_data = [0]
            self.integrated_g2_time = [0]
            self.integrated_g2_stage_pos = [self.current_stage_position]
            self.correlate(start=True, display_text=False, plot=False)
            self.int_g2_thread = Thread(target=self.integrated_g2)
            self.int_g2_thread.start()
        else:
            self.stop_intg2_thread = True
            # Stop correlation
            self.correlate(start=False, display_text=False)
            return

    def integrated_g2(self):
        ctr = 0
        acquistion_time = self.intg2_acqt.value()
        while True:
            if not self.stop_intg2_thread:

                # self.coincidences_vchannels = TimeTagger.Coincidence(self.tagger, channels=[1, 2], coincidenceWindow=1000, timestamp=TimeTagger.CoincidenceTimestamp.ListedFirst)

                # Clear the g2 after it's collected a many coincidences
                if np.sum(self.g2[0].getData()) > 1e7:
                    self.g2[0].clear()

                # Pause to acquire
                time.sleep(acquistion_time)

                # Get data from g2 integrated over all tau
                if ctr == 0:
                    data_old = np.sum(self.g2[0].getData())
                    data = data_old.copy()
                else:
                    data_new = np.sum(self.g2[0].getData())
                    data = data_new - data_old
                    data_old = data_new

                # data = np.sum(self.g2[0].getData())
                # self.g2[0].clear()
                data = data/acquistion_time

                singles_ch0 = self.cps[self.active_channels[0] - 1].value()
                singles_ch1 = self.cps[self.active_channels[1] - 1].value()
                mean_square_singles = np.sqrt(singles_ch0 * singles_ch1)

                if self.subtract_accidentals.isChecked():
                    data = data - self.accidentals.value()
                if self.divide_by_singles.isChecked():
                    # data = data / (np.sqrt(self.cps[0].value() * self.cps[1].value()))
                    data = 100 * data / mean_square_singles

                self.integrated_g2_data.append(data)

                new_time = self.integrated_g2_time[-1] + acquistion_time
                self.integrated_g2_time.append(new_time)
                self.integrated_g2_stage_pos.append(self.current_stage_position)

                self.coincidences.setValue(data)

                if self.divide_by_singles.isChecked():
                    self.coin_to_sing.setValue(data)
                else:
                    self.coin_to_sing.setValue(100 * data / mean_square_singles)

                n = int(self.intg2_npoints.value())
                ctr += 1
                if len(self.integrated_g2_data) > n:
                    self.intg2_graph.plot(t=self.integrated_g2_time[-n:-1], data=self.integrated_g2_data[-n:-1])
                    # self.intg2_graph.plot(t=self.integrated_g2_stage_pos[-n:-1], data=self.integrated_g2_data[-n:-1])
                else:
                    self.intg2_graph.plot(t=self.integrated_g2_time, data=self.integrated_g2_data)
                    # self.intg2_graph.plot(t=self.integrated_g2_stage_pos, data=self.integrated_g2_data)

                if ctr % n == 0:
                    print('trimmed')
                    self.integrated_g2_data = self.integrated_g2_data[-n:]
                    self.integrated_g2_time = self.integrated_g2_time[-n:]
                    self.integrated_g2_stage_pos = self.integrated_g2_stage_pos[-n:]
            else:
                break


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
    window = MainWindow() # Create an instance of our class
    app.exec_() # Start the application