import matplotlib.pyplot as plt
import SimpleITK as sitk


def myshow(img, zpos="default", title=None, margin=0.05, dpi=80):
  nda = sitk.GetArrayFromImage(img)
  spacing = img.GetSpacing()
  if zpos == "default":
    zpos = nda.shape[0] // 2
  print(zpos)

  if nda.ndim == 3:
    # fastest dim, either component or x
    c = nda.shape[-1]

    # the the number of components is 3 or 4 consider it an RGB image
    if c not in (3, 4):
      nda = nda[zpos, :, :]

  elif nda.ndim == 4:
    c = nda.shape[-1]

    if c not in (3, 4):
      raise RuntimeError("Unable to show 3D-vector Image")

    # take a z-slice
    nda = nda[zpos, :, :, :]

  xsize = nda.shape[1]
  ysize = nda.shape[0]

  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

  plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
  ax = plt.gca()

  extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

  t = ax.imshow(nda, extent=extent, interpolation=None)

  if nda.ndim == 2:
    t.set_cmap("gray")

  if title:
    plt.title(title)

  plt.show()


def myshow_with_seed(
  img, seed, zpos="default", title=None, margin=0.05, dpi=80
):
  nda = sitk.GetArrayFromImage(img)
  spacing = img.GetSpacing()
  if zpos == "default":
    zpos = nda.shape[0] // 2
  print(zpos)

  if nda.ndim == 3:
    # fastest dim, either component or x
    c = nda.shape[-1]

    # the the number of components is 3 or 4 consider it an RGB image
    if c not in (3, 4):
      nda = nda[zpos, :, :]

  elif nda.ndim == 4:
    c = nda.shape[-1]

    if c not in (3, 4):
      raise RuntimeError("Unable to show 3D-vector Image")

    # take a z-slice
    nda = nda[zpos, :, :, :]

  xsize = nda.shape[1]
  ysize = nda.shape[0]

  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

  plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
  ax = plt.gca()

  extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

  t = ax.imshow(nda, extent=extent, interpolation=None)
  ax.scatter(seed[0], seed[1], s=100, c="green")
  if nda.ndim == 2:
    t.set_cmap("gray")

  if title:
    plt.title(title)

  plt.show()


def mywrite(
  img, filename="default.png", zpos="default", title=None, margin=0.05, dpi=80
):
  nda = sitk.GetArrayFromImage(img)
  spacing = img.GetSpacing()
  if zpos == "default":
    zpos = nda.shape[0] // 2

  if nda.ndim == 3:
    # fastest dim, either component or x
    c = nda.shape[-1]

    # the the number of components is 3 or 4 consider it an RGB image
    if c not in (3, 4):
      nda = nda[zpos, :, :]

  elif nda.ndim == 4:
    c = nda.shape[-1]

    if c not in (3, 4):
      raise RuntimeError("Unable to show 3D-vector Image")

    # take a z-slice
    nda = nda[zpos, :, :, :]

  xsize = nda.shape[1]
  ysize = nda.shape[0]

  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

  plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
  ax = plt.gca()

  extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

  t = ax.imshow(nda, extent=extent, interpolation=None)

  if nda.ndim == 2:
    t.set_cmap("gray")

  if title:
    plt.title(title)

  # plt.show()
  plt.savefig(filename)


def mywrite_slice(
  img_slice,
  filename="default.png",
  zpos="default",
  title=None,
  margin=0.05,
  dpi=80,
  spacing=[1, 1],
):

  xsize = img_slice.shape[1]
  ysize = img_slice.shape[0]

  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

  plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
  ax = plt.gca()

  extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

  t = ax.imshow(img_slice, extent=extent, interpolation=None)

  if img_slice.ndim == 2:
    t.set_cmap("gray")

  # plt.show()
  plt.savefig(filename)

def mywrite_slice_subplots(
  img_slice,
  img2_slice,
  filename="default.png",
  zpos="default",
  title=None,
  margin=0.05,
  dpi=80,
  spacing=[1, 1],
):
  xsize = img_slice.shape[1]
  ysize = img_slice.shape[0]

  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

  fig, ax = plt.subplots(1,2,figsize=figsize, dpi=dpi, tight_layout=True)

  extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

  ax[0].imshow(img_slice, extent=extent, interpolation=None, cmap="gray")
  ax[1].imshow(img2_slice, extent=extent, interpolation=None, cmap="gray")
  # plt.show()
  plt.savefig(filename)

def mywrite_slice_with_seed(
  img_slice,
  seed,
  filename="default.png",
  zpos="default",
  title=None,
  margin=0.05,
  dpi=80,
  spacing=[1, 1],
):

  xsize = img_slice.shape[1]
  ysize = img_slice.shape[0]

  # Make a figure big enough to accommodate an axis of xpixels by ypixels
  # as well as the ticklabels, etc...
  figsize = (1 + margin) * xsize / dpi, (1 + margin) * ysize / dpi

  plt.figure(figsize=figsize, dpi=dpi, tight_layout=True)
  ax = plt.gca()

  extent = (0, xsize * spacing[0], ysize * spacing[1], 0)

  t = ax.imshow(img_slice, extent=extent, cmap="gray")
  ax.scatter(seed[0], seed[1], s=40, c="green")

  plt.savefig(filename)


def myshow3d(
  img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05, dpi=80
):
  img_xslices = [img[s, :, :] for s in xslices]
  img_yslices = [img[:, s, :] for s in yslices]
  img_zslices = [img[:, :, s] for s in zslices]

  maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))

  img_null = sitk.Image(
    [0, 0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel()
  )

  img_slices = []
  d = 0

  if len(img_xslices):
    img_slices += img_xslices + [img_null] * (maxlen - len(img_xslices))
    d += 1

  if len(img_yslices):
    img_slices += img_yslices + [img_null] * (maxlen - len(img_yslices))
    d += 1

  if len(img_zslices):
    img_slices += img_zslices + [img_null] * (maxlen - len(img_zslices))
    d += 1

  if maxlen != 0:
    if img.GetNumberOfComponentsPerPixel() == 1:
      img = sitk.Tile(img_slices, [maxlen, d])
    # TO DO check in code to get Tile Filter working with vector images
    else:
      img_comps = []
      for i in range(0, img.GetNumberOfComponentsPerPixel()):
        img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
        img_comps.append(sitk.Tile(img_slices_c, [maxlen, d]))
      img = sitk.Compose(img_comps)

  myshow(img, title, margin, dpi)
