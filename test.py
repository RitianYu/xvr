from diffdrr.pose import RigidTransform


identity = RigidTransform.identity().get_matrix()[0]
print(identity)