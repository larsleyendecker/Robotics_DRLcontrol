;; Auto-generated. Do not edit!


(when (boundp 'realsense2_camera::CustomVector)
  (if (not (find-package "REALSENSE2_CAMERA"))
    (make-package "REALSENSE2_CAMERA"))
  (shadow 'CustomVector (find-package "REALSENSE2_CAMERA")))
(unless (find-package "REALSENSE2_CAMERA::CUSTOMVECTOR")
  (make-package "REALSENSE2_CAMERA::CUSTOMVECTOR"))

(in-package "ROS")
;;//! \htmlinclude CustomVector.msg.html


(defclass realsense2_camera::CustomVector
  :super ros::object
  :slots (_data ))

(defmethod realsense2_camera::CustomVector
  (:init
   (&key
    ((:data __data) (make-array 0 :initial-element 0.0 :element-type :float))
    )
   (send-super :init)
   (setq _data __data)
   self)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; float32[] _data
    (* 4    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (sys::poke (elt _data i) (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate float-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :float)) (incf ptr- 4)
     ))
   ;;
   self)
  )

(setf (get realsense2_camera::CustomVector :md5sum-) "420cd38b6b071cd49f2970c3e2cee511")
(setf (get realsense2_camera::CustomVector :datatype-) "realsense2_camera/CustomVector")
(setf (get realsense2_camera::CustomVector :definition-)
      "float32[] data

")



(provide :realsense2_camera/CustomVector "420cd38b6b071cd49f2970c3e2cee511")


