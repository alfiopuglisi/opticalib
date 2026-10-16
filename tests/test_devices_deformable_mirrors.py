"""Tests for deformable mirror device interfaces."""

import socket
from unittest.mock import MagicMock, patch

import numpy as np

from opticalib.devices.deformable_mirrors import DP, NACODm


class TestDP:
    """Test the deformable platform interface."""

    @patch("opticalib.devices.deformable_mirrors._rc.get_iff_config", return_value=MagicMock())
    def test_read_buffer_organizes_data_by_diagnostic_key(self, mock_getiff):
        """Test that each buffer key contains a sample-by-actuator matrix."""
        sample_count = 4
        diagnostic_count = 17
        actuator_count = 111
        mock_getiff.return_value.get.return_value = {"frequency": 1.0}

        dm = DP.__new__(DP)
        dm._cmd_history = np.zeros((actuator_count, 2))
        dm._logger = MagicMock()
        dm._aoClient = MagicMock()

        subsystem = dm._aoClient.aoSystem.aoSubSystem0
        subsystem.sysConf.gen.cntFreq = 1000.0
        subsystem.support.diagBuf.read.return_value = {
            f"ch{actuator:04d}": (
                np.arange(sample_count * diagnostic_count).reshape(
                    sample_count, diagnostic_count
                )
                + actuator * 1000
            )
            for actuator in range(actuator_count)
        }

        with dm.read_buffer(segment=0, npoints_per_cmd=10) as result:
            assert result == {}

        assert len(result) == diagnostic_count
        assert result["actPos"].shape == (sample_count, actuator_count)
        np.testing.assert_array_equal(
            result["actPos"][:, 0],
            subsystem.support.diagBuf.read.return_value["ch0000"][:, 4],
        )
        np.testing.assert_array_equal(
            result["actPos"][:, -1],
            subsystem.support.diagBuf.read.return_value["ch0110"][:, 4],
        )
        np.testing.assert_array_equal(dm.bufferData["actPos"], result["actPos"])


class TestNACODm:
    """Test the NACO DM, which sends float32 commands to the PLC via UDP."""

    @staticmethod
    def _make_dm(monkeypatch, slave_ids=()):
        plc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        plc.bind(("127.0.0.1", 0))
        plc.settimeout(1)
        config = {"host": "127.0.0.1", "port": plc.getsockname()[1],
                  "slave_ids": list(slave_ids), "border_ids": []}
        monkeypatch.setattr(
            "opticalib.devices.deformable_mirrors._rc.get_device_config",
            lambda section, name: config,
        )
        dm = NACODm()
        plc.recvfrom(4096)  # zero command sent at startup
        return dm, plc

    def test_set_shape_sends_float32_packet(self, monkeypatch):
        dm, plc = self._make_dm(monkeypatch)
        cmd = np.arange(dm.n_acts) * 0.01
        dm.set_shape(cmd)
        data, _ = plc.recvfrom(4096)
        assert len(data) == dm.n_acts * 4
        np.testing.assert_allclose(np.frombuffer(data, "<f4"), cmd, rtol=1e-6)
        np.testing.assert_allclose(dm.get_shape(), cmd)
        dm.set_shape(cmd, differential=True)
        np.testing.assert_allclose(dm.get_shape(), 2 * cmd)

    def test_base_interface(self, monkeypatch):
        dm, _ = self._make_dm(monkeypatch, slave_ids=[0, 1])
        assert dm.n_acts == 185
        assert dm.act_coord.shape == (2, 185)
        assert dm.mirrorModes is None
        assert dm.cmd_history is None
        assert dm.slave_ids == [0, 1]
        assert dm.has_slaved_acts

    @patch("opticalib.devices.deformable_mirrors._rc.get_iff_config")
    def test_run_cmd_history_returns_to_start(self, mock_getiff, monkeypatch):
        mock_getiff.return_value = {"sequential_delay": 0.0}
        dm, plc = self._make_dm(monkeypatch)
        hist = np.zeros((dm.n_acts, 3))
        hist[5] = [1, -1, 1]
        dm.upload_cmd_history(hist)
        np.testing.assert_array_equal(dm.cmd_history, hist)
        sent = []
        tn = dm.run_cmd_history((lambda: sent.append(dm.get_shape()),))
        assert len(tn) == 3
        np.testing.assert_array_equal(np.array(sent).T, hist)
        np.testing.assert_array_equal(dm.get_shape(), np.zeros(dm.n_acts))
