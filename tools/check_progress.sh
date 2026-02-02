#!/bin/bash
# 翻译进度查看脚本

echo "======================================================================"
echo "           📚 深度学习书籍翻译进度监控"
echo "======================================================================"
echo ""

# 统计已完成文件数
COMPLETED=$(ls -1 chapters_chinese/ 2>/dev/null | wc -l | tr -d ' ')
TOTAL=130
PERCENT=$(echo "scale=1; $COMPLETED * 100 / $TOTAL" | bc)

echo "📊 翻译进度"
echo "   已完成: $COMPLETED / $TOTAL 文件 ($PERCENT%)"
echo ""

# 进度条
BARS=$(echo "scale=0; $COMPLETED * 50 / $TOTAL" | bc)
printf "   ["
for i in $(seq 1 $BARS); do printf "█"; done
for i in $(seq $BARS 49); do printf "░"; done
printf "] $PERCENT%%\n"
echo ""

# 输出大小
SIZE=$(du -sh chapters_chinese/ 2>/dev/null | cut -f1)
echo "💾 已翻译大小: $SIZE"
echo ""

# 最新3行进度
echo "🔄 最新进度:"
tail -3 /private/tmp/claude-502/-Users-a10093140-Desktop-me/tasks/bdacc72.output 2>/dev/null | sed 's/^/   /'
echo ""

# 预计完成时间
REMAINING=$((130 - COMPLETED))
EST_MIN=$(echo "scale=0; $REMAINING * 1.4" | bc)
EST_HOUR=$(echo "scale=1; $EST_MIN / 60" | bc)
echo "⏳ 预计剩余: 约 $EST_MIN 分钟 (~$EST_HOUR 小时)"
echo ""

echo "======================================================================"
echo "提示: 运行 'bash check_progress.sh' 查看最新进度"
echo "======================================================================"
