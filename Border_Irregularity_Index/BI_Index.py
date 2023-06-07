
def SmoothBoundary(x, thresh_width=5):
    '''
    Boundary smoothing module
    returns the smoothed groundtruth
    '''

    GS = GaussianSmoothing(1, (3, 3), 100, 2)
    CroporPad = tio.CropOrPad((1, 256, 256))
    smooth = torch.zeros(x.shape)
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
        # plt.figure()
        # plt.imshow(smooth[0][0].detach().numpy())
        contour_line = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1) - min_pool_x)
        # noyeaux = torch.nn.functional.relu(torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1))
        # x =plt.imshow(x[0 torch.nn.functional.relu(gs - contour_line)
        x = GS(contour_line)
        x_resized = CroporPad(x.unsqueeze(0))[0]
        x = (x > 0.2) * 1.00
        # x = torch.nn.functional.max_pool2d(x*-1, (2, 2), 1, 1)*-1
        # plt.imshow(x[0][0]*1.00)

    return x