# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from ur_dashboard_msgs/SafetyMode.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class SafetyMode(genpy.Message):
  _md5sum = "5da08725a63d4237bad689481131a84b"
  _type = "ur_dashboard_msgs/SafetyMode"
  _has_header = False  # flag to mark the presence of a Header object
  _full_text = """uint8 NORMAL=1
uint8 REDUCED=2
uint8 PROTECTIVE_STOP=3
uint8 RECOVERY=4
uint8 SAFEGUARD_STOP=5
uint8 SYSTEM_EMERGENCY_STOP=6
uint8 ROBOT_EMERGENCY_STOP=7
uint8 VIOLATION=8
uint8 FAULT=9
uint8 VALIDATE_JOINT_ID=10
uint8 UNDEFINED_SAFETY_MODE=11
uint8 AUTOMATIC_MODE_SAFEGUARD_STOP=12
uint8 SYSTEM_THREE_POSITION_ENABLING_STOP=13

uint8 mode
"""
  # Pseudo-constants
  NORMAL = 1
  REDUCED = 2
  PROTECTIVE_STOP = 3
  RECOVERY = 4
  SAFEGUARD_STOP = 5
  SYSTEM_EMERGENCY_STOP = 6
  ROBOT_EMERGENCY_STOP = 7
  VIOLATION = 8
  FAULT = 9
  VALIDATE_JOINT_ID = 10
  UNDEFINED_SAFETY_MODE = 11
  AUTOMATIC_MODE_SAFEGUARD_STOP = 12
  SYSTEM_THREE_POSITION_ENABLING_STOP = 13

  __slots__ = ['mode']
  _slot_types = ['uint8']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       mode

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(SafetyMode, self).__init__(*args, **kwds)
      # message fields cannot be None, assign default values for those that are
      if self.mode is None:
        self.mode = 0
    else:
      self.mode = 0

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      _x = self.mode
      buff.write(_get_struct_B().pack(_x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      end = 0
      start = end
      end += 1
      (self.mode,) = _get_struct_B().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      _x = self.mode
      buff.write(_get_struct_B().pack(_x))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      end = 0
      start = end
      end += 1
      (self.mode,) = _get_struct_B().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e)  # most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_B = None
def _get_struct_B():
    global _struct_B
    if _struct_B is None:
        _struct_B = struct.Struct("<B")
    return _struct_B
