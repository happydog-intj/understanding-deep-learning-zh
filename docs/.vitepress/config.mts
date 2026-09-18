import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  title: '理解深度学习',
  description: 'Understanding Deep Learning 中文翻译',
  lang: 'zh-CN',
  base: process.env.GITHUB_PAGES === 'true' ? '/understanding-deep-learning-zh/' : '/',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
  ],

  markdown: {
    config: (md) => {
      md.use(mathjax3)
    },
  },

  // MathJax styles need to be injected
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => tag.startsWith('mjx-'),
      },
    },
  },

  themeConfig: {
    logo: '/logo.svg',
    nav: [
      { text: '首页', link: '/' },
      { text: '开始阅读', link: '/preface' },
      {
        text: '章节',
        items: [
          { text: 'Part I 基础知识', link: '/chapters/01-introduction' },
          { text: 'Part II 模型训练', link: '/chapters/05-loss-functions' },
          { text: 'Part III 特殊架构', link: '/chapters/10-convolutional-networks' },
          { text: 'Part IV 生成模型', link: '/chapters/14-unsupervised-learning' },
          { text: 'Part V 强化学习与伦理', link: '/chapters/19-reinforcement-learning' },
          { text: '附录', link: '/appendices/a-notation' },
        ],
      },
    ],

    sidebar: [
      {
        text: '前言',
        items: [
          { text: '前言', link: '/preface' },
          { text: '致谢', link: '/acknowledgements' },
        ],
      },
      {
        text: 'Part I 基础知识',
        collapsed: false,
        items: [
          { text: '第1章 引言', link: '/chapters/01-introduction' },
          { text: '第2章 监督学习', link: '/chapters/02-supervised-learning' },
          { text: '第3章 浅层神经网络', link: '/chapters/03-shallow-neural-networks' },
          { text: '第4章 深度神经网络', link: '/chapters/04-deep-neural-networks' },
        ],
      },
      {
        text: 'Part II 模型训练',
        collapsed: false,
        items: [
          { text: '第5章 损失函数', link: '/chapters/05-loss-functions' },
          { text: '第6章 模型拟合', link: '/chapters/06-fitting-models' },
          { text: '第7章 梯度与初始化', link: '/chapters/07-gradients-and-initialization' },
          { text: '第8章 性能度量', link: '/chapters/08-measuring-performance' },
          { text: '第9章 正则化', link: '/chapters/09-regularization' },
        ],
      },
      {
        text: 'Part III 特殊架构',
        collapsed: false,
        items: [
          { text: '第10章 卷积网络', link: '/chapters/10-convolutional-networks' },
          { text: '第11章 残差网络', link: '/chapters/11-residual-networks' },
          { text: '第12章 Transformer', link: '/chapters/12-transformers' },
          { text: '第13章 图神经网络', link: '/chapters/13-graph-neural-networks' },
        ],
      },
      {
        text: 'Part IV 生成模型',
        collapsed: false,
        items: [
          { text: '第14章 无监督学习', link: '/chapters/14-unsupervised-learning' },
          { text: '第15章 生成对抗网络', link: '/chapters/15-generative-adversarial-networks' },
          { text: '第16章 归一化流', link: '/chapters/16-normalizing-flows' },
          { text: '第17章 变分自编码器', link: '/chapters/17-variational-autoencoders' },
          { text: '第18章 扩散模型', link: '/chapters/18-diffusion-models' },
        ],
      },
      {
        text: 'Part V 强化学习与伦理',
        collapsed: false,
        items: [
          { text: '第19章 强化学习', link: '/chapters/19-reinforcement-learning' },
          { text: '第20章 深度学习为何有效？', link: '/chapters/20-why-does-deep-learning-work' },
          { text: '第21章 深度学习与伦理', link: '/chapters/21-deep-learning-and-ethics' },
        ],
      },
      {
        text: '附录',
        collapsed: true,
        items: [
          { text: '附录A 符号表', link: '/appendices/a-notation' },
          { text: '附录B 数学基础', link: '/appendices/b-mathematics' },
          { text: '附录C 概率论基础', link: '/appendices/c-probability' },
        ],
      },
    ],

    outline: {
      level: [2, 3],
      label: '本章目录',
    },

    docFooter: {
      prev: '上一章',
      next: '下一章',
    },

    darkModeSwitchLabel: '主题',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '返回顶部',

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索',
            buttonAriaLabel: '搜索',
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭',
            },
          },
        },
      },
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/happydog-intj/understanding-deep-learning-zh' },
    ],

    footer: {
      message: '基于 CC-BY-NC-ND 许可协议',
      copyright: '原书版权 © Simon J.D. Prince · 翻译仅供学习交流',
    },
  },
})
