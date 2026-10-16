import numpy as _np
import socket
from opticalib.core.config import get_section_config
from opticalib.core.exceptions import CommandError
from opticalib.core import _types as _t


class NACOMirror:
    """
    Class to send commands to the NACO deformable mirror via PLC.

    The NACO mirror has no built-in position readback, so the last
    commanded vector is cached internally and returned by
    :meth:`get_shape`.

    Parameters
    ----------
    host : str
        IP address or hostname of the NACO mirror.
    port : int
        Port number for the PLC connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        """
        Initialize the NACO mirror interface.


        Parameters
        ----------
        host : str
            IP address or hostname of the NACO mirror.
        port : int
            Port number for the PLC connection.

        """
        self._host = host
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.n_acts = 185
        self._dmCoords = {
            "dm185": [7, 9, 11, 13, 15],
        }

        self._last_cmd: _t.ArrayLike = _np.zeros(self.n_acts)
        self._name = f"NACO{self.n_acts}"
        self.act_coord = self._init_act_coord()
        self._cmd_history = None
        self.mirrorModes = None
        self.refAct = None

    # ------------------------------------------------------------------
    # SDK-level interface
    # ------------------------------------------------------------------

    def get_shape(self) -> _t.ArrayLike:
        """
        Return the last commanded actuator positions.

        The NACO mirror does not provide a hardware readback; the last
        vector sent via :meth:`set_shape` is returned instead.

        Returns
        -------
        numpy.ndarray
            Array of length ``n_acts`` with the
            last commanded positions (zeros before the first command).
        """
        return self._last_cmd.copy()

    def set_shape(self, cmd: _t.ArrayLike) -> None:
        """
        Send an absolute command vector to the NACO mirror via the PLC.

        Data is sent as an UDP packet with 185 values in raw float32 format.
        The PLC will convert the values to the appropriate DAC units and send them to the mirror.

        Parameters
        ----------
        cmd : array_like
            Command vector of length ``n_acts``.

        Raises
        ------
        CommandError
            If the length of *cmd* does not match the number of
            actuators.
        """
        cmd = _np.asarray(cmd, dtype=float)
        if cmd.size != self.n_acts:
            raise CommandError(
                f"Command length {cmd.size} does not match the number "
                f"of actuators ({self.n_acts})."
            )
        # Send UDP packet to PLC (raw float32)
        self._sock.sendto(cmd.astype(_np.float32).tobytes(), (self._host, self._port))
        self._last_cmd = cmd.copy()

    # ------------------------------------------------------------------
    # Higher-level helpers
    # ------------------------------------------------------------------

    @property
    def n_actuators(self) -> int:
        """Number of actuators on the DM."""
        return self.n_acts

    # ------------------------------------------------------------------
    # Private initialisation helpers
    # ------------------------------------------------------------------

    def _init_act_coord(self) -> _t.ArrayLike:
        """
        Build the 2-D actuator coordinate array from the DM layout table.

        Returns
        -------
        numpy.ndarray or None
            Array of shape ``(2, n_acts)`` with ``(x, y)`` pixel
            coordinates, or an empty array if the model is unknown.
        """
        try:
            nacts_row_sequence = self._dmCoords[f"dm{self.n_acts}"]
        except KeyError:
            self.act_coord = _np.array([], dtype=int)
            return
        n_dim = nacts_row_sequence[-1]
        upper_rows = nacts_row_sequence[:-1]
        lower_rows = list(reversed(upper_rows))
        center_rows = [n_dim] * upper_rows[0]
        rows_number_of_acts = upper_rows + center_rows + lower_rows
        n_rows = len(rows_number_of_acts)
        cx = _np.array([], dtype=int)
        cy = _np.array([], dtype=int)
        for i in range(n_rows):
            cx = _np.concatenate(
                (
                    cx,
                    _np.arange(rows_number_of_acts[i])
                    + (n_dim - rows_number_of_acts[i]) // 2,
                )
            )
            cy = _np.concatenate((cy, _np.full(rows_number_of_acts[i], i)))
        self.act_coord = _np.array([cx, cy])
        return self.act_coord

