# 第12章：（Transformer）与第17章（变分自编码器）首次发表

*页码：14–14*

xii 目录  
汤姆·雅各布斯（Tom Jacobs）、雷·方（Lei Fang）、法比安·亨宁（Fabian Henning）、乌梅什·拉贾谢卡尔（Umesh Rajashekar）、杰伊·帕克（Jay Park）、凯·刘（Kai Liu）、巴勃罗·雷纳尔多·吉拉尔（Pablo Renard Guiral）、费德里科·巴尔贝罗（Federico Barbero）、荣江·潘（Rongjiang Pan）、贝廷·比尔坎·卡拉曼（Betin Bilkan Karaman）、莱奥尼达斯·瓦尔韦罗波洛斯（Leonidas Varveropoulos）、威廉·洛克四世（William Locke IV）、菲利普·贾西奥内克（Filip Jasionek）、王远航（Yuanhang Wang）、斯特凡·巴赫（Stefan Bach）、伊万·耶夫图申科（Ivan Yevtushenko）、大卫·格威尔（David Gwyer）、崔博涵（Bohan Cui）、阿里·达里亚尼（Ali Darijani）、鲁霍拉赫·法尔杭（Rouhollah Farhang）、唐力（Li Tang）、亚历山大·科塞列夫斯（Aleksandrs Koselevs）、王 Mason（Mason Wang）、巴勃罗·费尔南德斯（Pablo Fernandez）、安杰洛·科卢恰（Angelo Coluccia）、弗拉迪斯拉夫·莫罗尚（Vladyslav Moroshan）、拉米·卢伊斯托（Rami Luisto）、彼得·扎基（Peter Zaki）、卢卡斯·柯廷（Lucase Curtin）、维克多·刘（Victor Liu）、贾科莫·奇罗（Giacomo Cirò）、路易斯·内尔特纳（Louis Neltner）、艾哈迈德·切什梅奇（Ahmet Çeşmeci）、颜哲·贝克莫恩（Yanzhe Bekkemoen）、朱迪思·卡齐（Judith Katzy）以及扬内斯·布鲁恩斯（Jannes Bruns）。  

我特别感谢达尼亚尔·图尔穆哈姆别托夫（Daniyar Turmukhambetov）、阿梅代奥·布昂诺（Amedeo Buonanno）、安德烈亚·帕尼扎（Andrea Panizza）、马克·哈德森（Mark Hudson）、伯恩哈德·普法林格（Bernhard Pfahringer）、亚历山大·诺丁（Alexander Nordin）和尼古拉斯·洛德（Nicholas Lord），他们为本书多个章节提供了详尽的审阅意见。我尤其要感谢安德鲁·菲茨吉本（Andrew Fitzgibbon）、康斯坦蒂诺斯·德尔帕尼斯（Konstantinos Derpanis）、黑川俊昭（Toshiaki Kurokawa）和泰勒·米尔斯（Tyler Mills）——他们通读了全书，其热情极大地推动了本项目的完成。我也衷心感谢尼尔·坎贝尔（Neill Campbell）和厄兹居尔·希姆谢克（Özgür Şimşek），他们邀请我在巴斯大学（University of Bath）访学期间讲授了一门基于本书内容的课程，这是我首次以该材料开展教学。最后，我要向我的编辑伊丽莎白·斯威泽（Elizabeth Swayze）致以最诚挚的谢意，感谢她在整个出版过程中坦率而富有洞见的建议。  

第 12 章（Transformer）与第 17 章（变分自编码器）最初作为博文发表于 Borealis AI 官方博客；经授权，本文采用的改编版本由加拿大皇家银行（Royal Bank of Canada）与 Borealis AI 共同许可转载。我衷心感谢他们在本项目中给予的支持。第 16 章（归一化流）在结构上大致参考了 Kobyzev 等人（2020）所撰写的综述文章，本人亦为该文合著者之一。第 21 章有幸与达尔豪斯大学（Dalhousie University）的特拉维斯·拉克鲁瓦（Travis LaCroix）合作完成——他不仅合作愉快、轻松高效，更承担了该章绝大部分的撰写工作，令我深感幸运。  

版权声明  
• 图 1.13 中的棋盘图像改编自 http://tinyurl.com/yc2d54d4。  
• 图 1.2、1.4 和 1.10 中的齿轮图像改编自 http://tinyurl.com/2c7tttr8。  
• 图 19.1–19.5 及 19.6–19.9 中的企鹅图像改编自 http://tinyurl.com/ycx9je56。  
• 图 19.2–19.5、19.7、19.10–19.12 中的鱼图像改编自 http://tinyurl.com/4ueyhtsu。  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社（MIT Press）。