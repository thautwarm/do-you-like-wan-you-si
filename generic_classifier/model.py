from .utils import and_then, img_scale, img_redim, dump_model, load_model, read_directory
import numpy as np
import torch
import os
import re

def val(v):
    """
    compat lower version of Pytorch.
    """
    return torch.autograd.Variable(v, requires_grad=False)


class ConvStack(torch.nn.Module):
    def __init__(self, input_dims, pipeline=((12, 2), (8, 3), (16, 3))):
        super(ConvStack, self).__init__()
        filters, w, h = input_dims
        layers = []
        for (next_filters, stride) in pipeline:
            layers.append(torch.nn.Conv2d(filters, next_filters, stride))
            filters = next_filters
            w -= stride - 1
            h -= stride - 1
            layers.append(torch.nn.MaxPool2d(2, stride=2))
            w //= 2
            h //= 2
            layers.append(torch.nn.Dropout(0.3))

        self.layer = torch.nn.Sequential(*layers)
        self.dims = (filters, w, h)

    def forward(self, x):
        return self.layer(x)


class Mapper:
    def __init__(self, source):
        self.source = source

    def to_long(self, search_name):
        return next(long for name, long in self.source if name == search_name)

    def to_name(self, search_long):
        return next(name for name, long in self.source if long == search_long)


class SSP(torch.nn.Module):
    def __init__(self,
                 input_dims,
                 categories=2,
                 mapper: Mapper = None,
                 use_cuda=True):
        super(SSP, self).__init__()
        self.ssp = ConvStack(input_dims)
        filters, w, h = self.ssp.dims
        dim = filters * w * h
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(dim, dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim // 2, categories),
        )
        self.mapper = mapper
        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def set_mapper(self, mapper):
        self.mapper = mapper

    def forward(self, x):
        x = self.ssp(x)
        x = (lambda v: v.view(v.shape[0], -1))(x)
        x = self.predictor(x)
        return x

    def fit(self, samples, labels, lr=1e-3, epochs=100):
        labels = val(torch.from_numpy(labels).long())
        samples = val(torch.from_numpy(samples).float())
        if self.use_cuda:
            labels = labels.cuda()
            samples = samples.cuda()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print('Epoch [%d/%d], Loss: %s' % (epoch + 1, epochs, str(loss)))

        return loss.cpu().data.numpy().flatten()[0]

    def predict_with_names(self, samples):
        mapper = self.mapper
        return [((lambda x: x) if not mapper else mapper.to_name)(each)
                for each in self.predict(samples)]

    def predict(self, samples):
        samples = val(torch.from_numpy(samples).float())
        if self.use_cuda:
            samples = samples.cuda()
        return np.argmax(self(samples).cpu().data.numpy(), axis=1)

def make_data(X, y, size):
    """
    preprocess datasets and make batch-maker
    """
    _, w, h = size
    size = w, h
    X = np.array([img_redim(img_scale(x, size)) for x in X])
    X = X / 255.0

    def batch(batch_size=50):
        batch_size = min(batch_size, len(X))
        indices = np.random.permutation(len(X))
        while True:
            batch_indices = indices[:batch_size]
            yield X[batch_indices], y[batch_indices]
            np.random.shuffle(indices)

    return X, batch

def get_module(indir: str,
               outdir: str,
               size,
               md_name='mml',
               new=False,
               lr=0.0001,
               minor_epoch=50,
               batch_size=80,
               epoch=15):
    try:
        if new:
            raise Exception
        return load_model(f'{outdir}/{md_name}', map_location='cpu')
    except:
        ssp = None
        pass

    matcher = re.compile('cls_(.+)')

    cls = []
    X = []
    y = []
    for each in os.listdir(indir):
        m = matcher.match(each)
        if not m:
            continue
        name = m.groups()[0]

        long = len(cls)
        cls.append((name, long))
        x, y_ = read_directory(f'{indir}/cls_{name}', long)
        X += x
        y.append(y_)

    mapper = Mapper(cls)
    if ssp is None:
        ssp = SSP(size, categories=len(cls), mapper=mapper)

    print('training data loaded...')
    y = np.hstack(y)
    X, data_helper = make_data(X, y, size)
    i = 0
    print('data preprocessed.')
    try:
        for batch_x, batch_y in data_helper(batch_size):
            i += 1
            if i > epoch:
                break
            loss = ssp.fit(batch_x, batch_y, epochs=minor_epoch, lr=lr)
            if loss == .0:
                break
    except KeyboardInterrupt:
        pass
    dump_model(ssp, f'{outdir}/{md_name}')
    return ssp