"""Tests for opticalib.devices.interferometer static helper methods."""

from unittest.mock import MagicMock

import numpy as np
import numpy.ma as ma

from opticalib.devices.interferometer import _4DInterferometer, Wyko


class Test4DInterferometerStatics:
    """Tests for static methods in _4DInterferometer."""

    def test_get_camera_settings_uses_new_width_method(self, monkeypatch):
        """Test camera settings retrieval with the renamed width accessor."""

        class ReaderStub:
            """Reader stub exposing only the new width API."""

            def get_image_width_in_pixels(self):
                return 2000

            def get_image_height_in_pixels(self):
                return 1500

            def get_offset_x(self):
                return 11

            def get_offset_y(self):
                return 22

        monkeypatch.setattr(
            "opticalib.devices.interferometer._confReader",
            lambda _path: ReaderStub(),
        )

        settings = _4DInterferometer.get_camera_settings()

        assert settings == [2000, 1500, 11, 22]

    def test_get_frame_rate_reads_from_reader(self, monkeypatch):
        """Test frame-rate retrieval through the configured reader."""

        class ReaderStub:
            """Reader stub for frame-rate retrieval."""

            def get_frame_rate(self):
                return 77.5

        monkeypatch.setattr(
            "opticalib.devices.interferometer._confReader",
            lambda _path: ReaderStub(),
        )

        frame_rate = _4DInterferometer.get_frame_rate()

        assert frame_rate == 77.5

    def test_into_full_frame_uses_camera_offsets_when_offset_is_none(
        self, monkeypatch
    ):
        """Test that into_full_frame works with default optional offset."""
        monkeypatch.setattr(
            "opticalib.devices.interferometer._4DInterferometer.get_camera_settings",
            staticmethod(lambda _tn=None: [2048, 2048, 4, 7]),
        )

        img = ma.masked_array(np.array([[1.0, 2.0], [3.0, 4.0]]))

        fullimg = _4DInterferometer.into_full_frame(img)

        assert fullimg.shape == (2048, 2048)
        np.testing.assert_array_equal(fullimg[7:9, 4:6], img.data)


    def test_into_full_frame_reads_offsets_from_config_path(self, monkeypatch):
        """Test config_path-based offset loading for full-frame insertion."""
        reader = MagicMock()
        reader.get_offset_x.return_value = 0
        reader.get_offset_y.return_value = 0
        monkeypatch.setattr(
            "opticalib.devices.interferometer._fn.ConfSettingReader4D",
            lambda _path: reader,
        )

        img = ma.masked_array(np.array([[3.0, 1.0]]))

        fullimg = _4DInterferometer.into_full_frame(
            img,
            config_path="/tmp/fake_4d_settings.ini",
        )

        np.testing.assert_array_equal(fullimg[0:1, 0:2], img.data)


class TestWyko:
    """Tests for the Wyko, reached through the plico_interferometer client."""

    @staticmethod
    def _make_wyko(monkeypatch):
        client = MagicMock()
        client.wavefront.side_effect = lambda n=1: ma.masked_array(
            np.full((4, 4), float(n)), mask=np.eye(4, dtype=bool)
        )
        factory = MagicMock(return_value=client)
        monkeypatch.setattr(
            "plico_interferometer.interferometer_4SightFocus_client", factory
        )
        monkeypatch.setattr(
            "opticalib.core.config.get_section_config",
            lambda *_: {"Wyko4100": {"ip": "10.0.0.1", "port": "8011"}},
        )
        return Wyko(), factory, client

    def test_connects_with_configured_address(self, monkeypatch):
        wyko, factory, _ = self._make_wyko(monkeypatch)
        factory.assert_called_once_with("10.0.0.1", 8011)
        assert wyko._name == "Wyko4100"

    def test_acquire_map(self, monkeypatch):
        wyko, _, client = self._make_wyko(monkeypatch)
        img = wyko.acquire_map(nframes=3)
        client.wavefront.assert_called_once_with(3)
        assert img.shape == (4, 4)
        assert img.mask[0, 0] and not img.mask[0, 1]
        img = wyko.acquire_map(nframes=2, delay=1e-3)
        assert client.wavefront.call_count == 3
        assert np.allclose(img.compressed(), 1.0)

    def test_server_mode(self, monkeypatch):
        client = MagicMock()
        client.wavefront.side_effect = lambda n=1: ma.masked_array(
            np.full((4, 4), 1.0), mask=np.eye(4, dtype=bool)
        )
        factory = MagicMock(return_value=client)
        monkeypatch.setattr("plico_interferometer.interferometer", factory)
        monkeypatch.setattr(
            "opticalib.core.config.get_section_config",
            lambda *_: {
                "Wyko4100": {"ip": "10.0.0.1", "port": 7300, "server": True}
            },
        )
        wyko = Wyko()
        factory.assert_called_once_with("10.0.0.1", 7300)
        assert wyko.server
        # the server returns one frame per call: averaging is done here
        img = wyko.acquire_map(nframes=3)
        assert client.wavefront.call_count == 3
        assert np.allclose(img.compressed(), 1.0)
