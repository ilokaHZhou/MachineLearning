# 环境管理工具

| 工具             | 优点                       | 缺点                   | 使用场景            |
| -----------------| -------------------------- | --------------------- | ------------------- |
| **virtualenv**  | python2时代广泛使用         | 需要手动管理虚拟环境和依赖 | 不推荐            |
| **venv**         | 内置于Python，轻量简单      | 没有依赖管理功能，不会自动解决包的依赖冲突，需要手动使用requirements.txt | 只需要轻量化虚拟环境的时候，自己手搓依赖管理吧 |
| **pipenv**       | 自动管理环境和依赖，兼容性好，安装包小（只包含所需依赖） | 无法管理非Python的包，大型项目的环境管理效率不高 | 好用，爱用，中小型项目闭眼直接用 |
| **Anaconda**       | 预装大量数据科学和机器学习工具库<br />支持Python和非Python包<br />提供图形化界面，开箱即用<br />兼容PyCharm | 安装包巨大<br />package源是Anaconda Repository而不是PyPI官方，更新和维护较慢 | 适合数据科学和机器学习，不适合普通项目开发 |
| **miniconda**       | 轻量化的Anaconda，不预装库 | 需要手动安装库，初始配置工作多，package源不是PyPI | 轻量化的自定义环境 |

![venn_diagram](./assets/venn_diagram.png "venn_diagram")^[https://alpopkes.com/posts/python/packaging_tools/]

**virtualenv**: 太老，Python2时的工具，不推荐。

**venv**: python自带的虚拟环境管理，简单但功能少。

只能创建虚拟环境，不能指定系统没有的python环境版本，不能管理系统中的环境列表（例如选择一个已经创建好了的虚拟环境）。venv的虚拟环境默认是存放在项目文件夹里的，这会影响项目文件的管理。

**pipenv**: requests库作者Kenneth Reitz大神的作品。上手轻松，兼容`vitualenv`和`venv`（`pipenv install`实际上是起了一个virtualenv环境，用`venv`创建的环境也可以在激活后使用`pipenv`来管理依赖）使用Pipefile和Pipefile.lock管理版本和依赖，可以读取项目目录中已经创建的`requirements.txt`，运行install时会检查`Pipfile.lock`已经存在的依赖，避免两个包有相同依赖时重复安装依赖。

但`pipenv`并不稳定，例如，如果你运行`pip install`你要装的包两次，结果可能不一样，pipenv曾承诺解决这个问题，但实际上，它只是多次尝试运行`pip install <单个包>`，直到结果看起来差不多符合规范。

**Anaconda / conda**: 如果是科学计算的新手，推荐使用，会自带spyder，jupyternotebook等包，但：

anaconda过于臃肿，它的安装包里包括了众多科学计算会用到的packages，安装后动辄5-6个G。
anaconda有个不包含packages的版本，叫miniconda，但miniconda仍然存在安装依赖库过于激进的问题，安装同样的packages，conda总会比别的包管理器安装更多的“依赖包”，即便有的“依赖包”并不是必须，这会导致你的项目出现不必要的膨胀。
同时，conda的packages列表“conda list”还存在和“pip list”不一致的问题。

**poetry**: 摸索中...

poetry使用pyproject.toml 和 poetry.lock文件来管理依赖，类似于JavaScript/Node.js的Npm和Rust的Cargo，这俩都是非常成熟好用的依赖管理方案。
poetry本身并不具有管理Python解释器的功能，推荐和pyenv/pyenv-win使用，可以轻松下载和设置不同版本的Python解释器。^[https://zhuanlan.zhihu.com/p/663735038]

**rye**：摸索中...

