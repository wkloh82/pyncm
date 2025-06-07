# -*- coding: utf-8 -*-
# PyNCM CLI interface

from pyncm import (
    DumpSessionAsString,
    GetCurrentSession,
    LoadSessionFromString,
    SetCurrentSession,
    __version__,
)
from pyncm.utils.lrcparser import LrcParser
from pyncm.utils.yrcparser import YrcParser, ASSWriter, YrcLine, YrcBlock
from pyncm.utils.helper import (
    TrackHelper,
    ArtistHelper,
    UserHelper,
    FuzzyPathHelper,
    SubstituteWithFullwidth,
)
from pyncm.apis import artist, login, track, playlist, album, user
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from time import sleep
from os.path import join, exists
from os import remove, makedirs
from dataclasses import dataclass

from logging import exception, getLogger, basicConfig
import sys, argparse, re, os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QGroupBox, QFormLayout, QComboBox, QTableWidget, QMessageBox,
    QFileDialog, QTableWidgetItem, QDialog
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineProfile
import json
from pathlib import Path

logger = getLogger("pyncm.main")
# Import checks
OPTIONALS = {"mutagen": False, "tqdm": False, "coloredlogs": False}
OPTIONALS_MISSING_INFO = {
    "mutagen": "无法为下载的音乐添加歌手信息，封面等资源",
    "tqdm": "将不会显示下载进度条",
    "coloredlogs": "日志不会以彩色输出",
}
from importlib.util import find_spec

for import_name in OPTIONALS:
    OPTIONALS[import_name] = find_spec(import_name)
    if not OPTIONALS[import_name]:
        sys.stderr.writelines(
            [f"[WARN] {import_name} 没有安装，{OPTIONALS_MISSING_INFO[import_name]}\n"]
        )

__desc__ = """PyNCM 网易云音乐下载工具 %s""" % __version__


class TaskPoolExecutorThread(Thread):
    @staticmethod
    def tag_audio(track: TrackHelper, file: str, cover_img: str = ""):
        if not OPTIONALS["mutagen"]:
            return

        def write_keys(song):
            # Writing metadata
            # Due to different capabilites of containers, only
            # ones that can actually be stored will be written.
            complete_metadata = {
                "title": [track.TrackName],
                "artist": [*(track.Artists or [])],
                "albumartist": [*(track.Album.AlbumArtists or [])],
                "album": [track.AlbumName],
                "tracknumber": "%s/%s"
                % (track.TrackNumber, track.Album.AlbumSongCount),
                "date": [
                    str(track.Album.AlbumPublishTime)
                ],  # TrackPublishTime is not very reliable!
                "copyright": [track.Album.AlbumCompany or ""],
                "discnumber": [track.CD],
                # These only applies to vorbis tags (e.g. FLAC)
                "totaltracks": [str(track.Album.AlbumSongCount)],
                "ncm-id": [str(track.ID)],
            }
            for k, v in complete_metadata.items():
                try:
                    song[k] = v
                except:
                    pass
            song.save()

        def mp4():
            from mutagen import easymp4
            from mutagen.mp4 import MP4, MP4Cover

            song = easymp4.EasyMP4(file)
            write_keys(song)
            if exists(cover_img):
                song = MP4(file)
                song["covr"] = [MP4Cover(open(cover_img, "rb").read())]
                song.save()

        def mp3():
            from mutagen.mp3 import EasyMP3, HeaderNotFoundError
            from mutagen.id3 import ID3, APIC

            try:
                song = EasyMP3(file)
            except HeaderNotFoundError:
                song = EasyMP3()
                song.filename = file
                song.save()
                song = EasyMP3(file)
            write_keys(song)
            if exists(cover_img):
                song = ID3(file)
                song.update_to_v23()  # better compatibility over v2.4
                song.add(
                    APIC(
                        encoding=3,
                        mime="image/jpeg",
                        type=3,
                        desc="",
                        data=open(cover_img, "rb").read(),
                    )
                )
                song.save(v2_version=3)

        def flac():
            from mutagen.flac import FLAC, Picture
            from mutagen.mp3 import EasyMP3

            song = FLAC(file)
            write_keys(song)
            if exists(cover_img):
                pic = Picture()
                pic.data = open(cover_img, "rb").read()
                pic.mime = "image/jpeg"
                song.add_picture(pic)
                song.save()

        def ogg():
            import base64
            from mutagen.flac import Picture
            from mutagen.oggvorbis import OggVorbis

            song = OggVorbis(file)
            write_keys(song)
            if exists(cover_img):
                pic = Picture()
                pic.data = open(cover_img, "rb").read()
                pic.mime = "image/jpeg"
                song["metadata_block_picture"] = [
                    base64.b64encode(pic.write()).decode("ascii")
                ]
                song.save()

        format = file.split(".")[-1].upper()
        for ext, method in [
            ({"M4A", "M4B", "M4P", "MP4"}, mp4),
            ({"MP3"}, mp3),
            ({"FLAC"}, flac),
            ({"OGG", "OGV"}, ogg),
        ]:
            if format in ext:
                return method() or True
        return False

    def download_by_url(self, url, dest, xfer=False):
        # Downloads generic content
        response = GetCurrentSession().get(url, stream=True)
        length = int(response.headers.get("content-length"))

        with open(dest, "wb") as f:
            for chunk in response.iter_content(128 * 2**10):
                self.xfered += len(chunk)
                if xfer:
                    self.finished_tasks += len(chunk) / length  # task [0,1]
                f.write(chunk)  # write every 128KB read
        return dest

    def __init__(self, *a, max_workers=4, **k):
        super().__init__(*a, **k)
        self.daemon = True
        self.finished_tasks: float = 0
        self.xfered = 0
        self.task_queue = Queue()
        self.max_workers = max_workers

    def run(self):
        def execute(task):
            if type(task) == TrackDownloadTask:
                try:
                    if task.skip_download:
                        return
                    # Downloding source audio
                    apiCall = (
                        track.GetTrackAudioV1
                        if not task.routine.args.use_download_api
                        else track.GetTrackDownloadURLV1
                    )
                    if task.routine.args.use_download_api:
                        logger.warning("使用下载 API，可能消耗 VIP 下载额度！")
                    dAudio = apiCall(task.audio.id, level=task.audio.level)
                    assert "data" in dAudio, "其他错误： %s" % dAudio
                    dAudio = dAudio["data"]
                    if type(dAudio) == list:
                        dAudio = dAudio[0]
                    if not dAudio["url"]:
                        # Attempt to give some sort of explaination
                        # 来自 https://neteasecloudmusicapi-docs.4everland.app/#/?id=%e8%8e%b7%e5%8f%96%e6%ad%8c%e6%9b%b2%e8%af%a6%e6%83%85
                        """fee : enum
                        0: 免费或无版权
                        1: VIP 歌曲
                        4: 购买专辑
                        8: 非会员可免费播放低音质，会员可播放高音质及下载
                        fee 为 1 或 8 的歌曲均可单独购买 2 元单曲
                        """
                        fee = dAudio["fee"]
                        assert fee != 0, "可能无版权"
                        assert fee != 1, "VIP歌曲，账户可能无权访问"
                        assert fee != 4, "歌曲所在专辑需购买"
                        assert fee != 8, "歌曲可能需要单独购买或以低音质加载"
                        assert False, "未知原因 (fee=%d)" % fee
                    logger.info(
                        "开始下载 #%d / %d - %s - %s - %skbps - %s"
                        % (
                            task.index + 1,
                            task.total,
                            task.song.Title,
                            task.song.AlbumName,
                            dAudio["br"] // 1000,
                            dAudio["type"].upper(),
                        )
                    )
                    task.extension = dAudio["type"].lower()
                    if not exists(task.audio.dest):
                        makedirs(task.audio.dest)
                    dest_src = self.download_by_url(
                        dAudio["url"],
                        task.save_as + "." + dAudio["type"].lower(),
                        xfer=True,
                    )
                    # Downloading cover
                    dest_cvr = self.download_by_url(
                        task.cover.url, task.save_as + ".jpg"
                    )
                    # Downloading & Parsing lyrics
                    lrc = LrcParser()
                    dLyrics = track.GetTrackLyricsNew(task.lyrics.id)
                    for k in set(dLyrics.keys()) & (
                        {"lrc", "tlyric", "romalrc"} - task.lyrics.lrc_blacklist
                    ):  # Filtering LRCs
                        lrc.LoadLrc(dLyrics[k]["lyric"])
                    lrc_text = lrc.DumpLyrics()
                    if lrc_text:
                        open(task.save_as + ".lrc", "w", encoding="utf-8").write(
                            lrc_text
                        )
                    # `yrc` (whatever that means) lyrics contains syllable-by-syllable time sigs
                    if not "yrc" in task.lyrics.lrc_blacklist and "yrc" in dLyrics:
                        yrc = YrcParser(
                            dLyrics["yrc"]["version"], dLyrics["yrc"]["lyric"]
                        )
                        parsed = yrc.parse()
                        writer = ASSWriter()

                        for line in parsed:
                            line: YrcLine
                            writer.begin_line(line.t_begin, line.t_end)
                            for block in line:
                                block: YrcBlock
                                if block.meta:
                                    writer.add_meta(YrcParser.extract_meta(block.meta))
                                else:
                                    writer.add_syllable(block.t_duration, block.text)
                            writer.end_line()

                        open(task.save_as + ".ass", "w", encoding="utf-8").write(
                            writer.content
                        )
                    # Tagging the audio
                    try:
                        self.tag_audio(task.song, dest_src, dest_cvr)
                    except Exception as e:
                        logger.warning("标签失败 - %s - %s" % (task.song.Title, e))
                    logger.info(
                        "完成下载 #%d / %d - %s"
                        % (task.index + 1, task.total, task.song.Title)
                    )
                except Exception as e:
                    logger.warning("下载失败 %s - %s" % (task.song.Title, e))
                    task.routine.result_exception(task.song.ID, e, task.song.Title)
                # Cleaning up
                remove(dest_cvr)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while True:
                task = self.task_queue.get()
                future = executor.submit(execute, task)
                future.add_done_callback(lambda future: self.task_queue.task_done())


# Subroutines
class Subroutine:
    """Generic subroutine

    Subroutines are `callable`,upon called with `ids`,one
    queues tasks with all given arguments via `put_func` callback

    `prefix` is used to identify subroutines,especially when one is a child
    of another subroutine.
    """

    exceptions = None

    def __init__(self, args, put_func, prefix=None) -> None:
        self.args = args
        self.put = put_func
        self.prefix = prefix or self.prefix
        self.exceptions = dict()

    def result_exception(self, result_id, exception: Exception, desc=None):
        self.exceptions.setdefault(result_id, list())
        self.exceptions[result_id].append((exception, desc))

    @property
    def has_exceptions(self):
        return len(self.exceptions) > 0


@dataclass
class BaseDownloadTask:
    id: int = 0
    url: str = ""
    dest: str = ""
    level: str = ""


@dataclass
class LyricsDownloadTask(BaseDownloadTask):
    id: int = 0
    dest: str = ""
    lrc_blacklist: set = None


@dataclass
class TrackDownloadTask:
    song: TrackHelper = None
    cover: BaseDownloadTask = None
    lyrics: BaseDownloadTask = None
    audio: BaseDownloadTask = None

    index: int = 0
    total: int = 0
    lyrics_exclude: set = None
    save_as: str = ""
    extension: str = ""

    routine: Subroutine = None

    skip_download: bool = False


class Playlist(Subroutine):
    prefix = "歌单"
    """Base routine for ID-based tasks"""

    def filter(self, song_list):
        # This is only meant to faciliate sorting w/ APIs that doesn't implement them
        # The observed behaviors remains more or less the same with the offical ones
        if self.args.count > 0 and self.args.sort_by != "default":
            sorting = {
                "hot": lambda song: float(song["pop"]),  # [0,100.0]
                "time": lambda song: TrackHelper(
                    song
                ).Album.AlbumPublishTime,  # in Years
            }[self.args.sort_by]
            song_list = sorted(
                song_list, key=sorting, reverse=not self.args.reverse_sort
            )
        return song_list[: self.args.count if self.args.count > 0 else len(song_list)]

    def forIds(self, ids):
        dDetails = [
            track.GetTrackDetail(ids[index : min(len(ids), index + 1000)]).get("songs")
            for index in range(0, len(ids), 1000)
        ]
        dDetails = [song for stripe in dDetails for song in stripe]
        dDetails = self.filter(dDetails)
        downloadTasks = []
        index = 0
        for index, dDetail in enumerate(dDetails):
            try:
                song = TrackHelper(dDetail)
                output_name = SubstituteWithFullwidth(
                    self.args.output_name.format(**song.template)
                )
                output_folder = self.args.output.format(
                    **{k: SubstituteWithFullwidth(v) for k, v in song.template.items()}
                )

                tSong = TrackDownloadTask(
                    index=index,
                    total=len(dDetails),
                    song=song,
                    cover=BaseDownloadTask(
                        id=song.ID, url=song.AlbumCover, dest=output_folder
                    ),
                    audio=BaseDownloadTask(
                        id=song.ID,
                        level=self.args.quality,
                        dest=output_folder,
                    ),
                    lyrics=LyricsDownloadTask(
                        id=song.ID,
                        dest=output_folder,
                        lrc_blacklist=set(self.args.lyric_no),
                    ),
                    save_as=join(output_folder, output_name),
                    routine=self,
                )
                downloadTasks.append(tSong)
                # If audio file already exsists
                # Skip its download if `--no_overwrite` is explicitly set
                if self.args.no_overwrite:
                    if FuzzyPathHelper(output_folder).exists(
                        output_name, partial_extension_check=True
                    ):
                        logger.warning(
                            "单曲 #%d / %d - %s - %s 已存在，跳过"
                            % (index + 1, len(dDetails), song.Title, song.AlbumName)
                        )
                        tSong.skip_download = True
                        tSong.extension = FuzzyPathHelper(output_folder).get_extension(
                            output_name
                        )
                        self.put(tSong)
                        continue
                self.put(tSong)

            except Exception as e:
                logger.warning(
                    "单曲 #%d / %d - %s - %s 无法下载： %s"
                    % (index + 1, len(dDetails), song.Title, song.AlbumName, e)
                )
                self.result_exception(song.ID, e, song.Title)
        return downloadTasks

    def __call__(self, ids):
        queued = []
        for _id in ids:
            dList = playlist.GetPlaylistInfo(_id)
            logger.info(self.prefix + "：%s" % dict(dList)["playlist"]["name"])
            queuedTasks = self.forIds(
                [tid.get("id") for tid in dict(dList)["playlist"]["trackIds"]]
            )
            queued += queuedTasks
        return queued


class Album(Playlist):
    prefix = "专辑"

    def __call__(self, ids):
        queued = []
        for _id in ids:
            dList = album.GetAlbumInfo(_id)
            logger.info(self.prefix + "：%s" % dict(dList)["album"]["name"])
            queuedTasks = self.forIds([tid["id"] for tid in dList["songs"]])
            queued += queuedTasks
        return queued


class Artist(Playlist):
    prefix = "艺术家"

    def __call__(self, ids):
        queued = []
        for _id in ids:
            logger.info(self.prefix + "：%s" % ArtistHelper(_id).ArtistName)
            # dList = artist.GetArtisTracks(_id,limit=self.args.count,order=self.args.sort_by)
            # This API is rather inconsistent for some reason. Sometimes 'songs' list
            # would be straight out empty
            # TODO: Fix GetArtistTracks
            # We iterate all Albums instead as this would provide a superset of what `GetArtistsTracks` gives us
            album_ids = [
                album["id"] for album in artist.GetArtistAlbums(_id)["hotAlbums"]
            ]
            album_task = Album(self.args, self.put, prefix="艺术家专辑")
            album_task.forIds = self.forIds
            # All exceptions will still be handled by this subroutine
            # TODO: Try to... handle this nicer?
            queued += album_task(album_ids)
        return queued


class User(Playlist):
    prefix = "用户"

    def __call__(self, ids):
        queued = []
        for _id in ids:
            logger.info(self.prefix + "： %s" % UserHelper(_id).UserName)
            logger.warning(
                "同时下载收藏歌单"
                if self.args.user_bookmarks
                else "只下载该用户创建的歌单"
            )
            playlist_ids = [
                pl["id"]
                for pl in user.GetUserPlaylists(_id)["playlist"]
                if self.args.user_bookmarks
                or pl["creator"]["userId"] == UserHelper(_id).ID
            ]
            playlist_task = Playlist(self.args, self.put, prefix="用户歌单")
            playlist_task.forIds = self.forIds
            queued += playlist_task(playlist_ids)
        return queued


class Song(Playlist):
    def __call__(self, ids):
        queuedTasks = self.forIds(ids)
        if self.args.save_m3u:
            logger.warning("不能为单曲保存 m3u 文件")
        return queuedTasks


def create_subroutine(sub_type) -> Subroutine:
    """Dynamically creates subroutine callable by string specified"""
    return {
        "song": Song,
        "playlist": Playlist,
        "album": Album,
        "artist": Artist,
        "user": User,
    }[sub_type]


def parse_sharelink(url):
    """Parses (partial) URLs for NE resources and determines its ID and type"""
    try:
        rurl = re.findall(r"(?:http|https):\/\/.*", url)
        if rurl:
            url = rurl[0]  # Use first URL found. Otherwise use value given as is.
        numerics = re.findall(r"\d{4,}", url)
        if not numerics:
            raise ValueError("未在链接中找到任何 ID")
            
        ids = numerics[:1]  # Only pick the first match
        table = {
            "song": ["trackId", "song"],
            "playlist": ["playlist"],
            "artist": ["artist"],
            "album": ["album"],
            "user": ["user"],
        }
        rtype = "song"  # Defaults to songs (tracks)
        best_index = len(url)
        
        for rtype_, rkeyword in table.items():
            for kw in rkeyword:
                try:
                    index = url.index(kw)
                    if index < best_index:
                        best_index = index
                        rtype = rtype_
                except ValueError:
                    continue
                    
        return rtype, ids
        
    except Exception as e:
        logger.exception(f"解析链接失败: {url}")
        raise ValueError(f"解析链接失败: {str(e)}")


PLACEHOLDER_URL = "00000"


def parse_args(quit_on_empty_args=True):
    """Setting up __main__ argparser"""
    parser = argparse.ArgumentParser(
        description=__desc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "url",
        metavar="链接",
        help="网易云音乐分享链接",
        nargs="*",
        default=PLACEHOLDER_URL,
    )
    group = parser.add_argument_group("下载")
    group.add_argument(
        "--max-workers", "-m", metavar="最多同时下载任务数", default=4, type=int
    )
    group.add_argument(
        "--output-name",
        "--template",
        "-t",
        metavar="保存文件名模板",
        help=r"""保存文件名模板
    参数：    
        id     - 网易云音乐资源 ID
        year   - 出版年份
        no     - 专辑中编号
        album  - 专辑标题
        track  - 单曲标题        
        title  - 完整标题
        artists- 艺术家名
    例：
        {track} - {artists} 等效于 {title}""",
        default=r"{title}",
    )
    group.add_argument(
        "-o",
        "--output",
        metavar="输出",
        default=".",
        help=r"""输出文件夹
    注：该参数也可使用模板，格式同 保存文件名模板""",
    )
    group.add_argument(
        "--quality",
        metavar="音质",
        choices=["standard", "exhigh", "lossless", "hires"],
        help=r"""音频音质（高音质需要 CVIP）
    参数：
        hires  - Hi-Res
        lossless- "无损"
        exhigh  - 较高
        standard- 标准""",
        default="standard",
    )
    group.add_argument(
        "-dl",
        "--use-download-api",
        action="store_true",
        help="调用下载API，而非播放API进行下载。如此可能允许更高高音质音频的下载。\n【注意】此API有额度限制，参考 https://music.163.com/member/downinfo",
    )
    group.add_argument(
        "--no-overwrite", action="store_true", help="不重复下载已经存在的音频文件"
    )
    group = parser.add_argument_group("歌词")
    group.add_argument(
        "--lyric-no",
        metavar="跳过歌词",
        help=r"""跳过某些歌词类型的下载
    参数：        
        lrc    - 源语言歌词  (合并到 .lrc)
        tlyric - 翻译后歌词  (合并到 .lrc)
        romalrc- 罗马音歌词  (合并到 .lrc)
        yrc    - 逐词滚动歌词 (保存到 .ass)
        none   - 下载所有歌词
    例：
        --lyric-no "tlyric romalrc yrc" 将只下载源语言歌词
        --lyric-no none 将下载所有歌词
    注：
        默认不下载 *逐词滚动歌词*
        """,
        default="yrc",
    )
    group = parser.add_argument_group("登陆")
    group.add_argument("--phone", metavar="手机", default="", help="网易账户手机号")
    group.add_argument(
        "--cookie",
        metavar="Cookie (MUSIC_U)",
        default="",
        help="网易云音乐 MUSIC_U Cookie (形如 '00B2471D143...')",
    )
    group.add_argument(
        "--pwd", "--password", metavar="密码", default="", help="网易账户密码"
    )
    group.add_argument(
        "--save", metavar="[保存到]", default="", help="写本次登录信息于文件"
    )
    group.add_argument(
        "--load",
        metavar="[保存的登陆信息文件]",
        default="",
        help="从文件读取登录信息供本次登陆使用",
    )
    group.add_argument(
        "--http", action="store_true", help="优先使用 HTTP，不保证不被升级"
    )
    group.add_argument(
        "--deviceId",
        metavar="设备ID",
        default="",
        help="指定设备 ID；匿名登陆时，设备 ID 既指定对应账户\n【注意】默认 ID 与当前设备无关，乃从内嵌 256 可用 ID 中随机选取；指定自定义 ID 不一定能登录，相关性暂时未知",
    )
    group.add_argument("--log-level", help="日志等级", default="NOTSET")
    group = parser.add_argument_group("限量及过滤（注：只适用于*每单个*链接 / ID）")
    group.add_argument(
        "-n",
        "--count",
        metavar="下载总量",
        default=0,
        help="限制下载歌曲总量，n=0即不限制（注：过大值可能导致限流）",
        type=int,
    )
    group.add_argument(
        "--sort-by",
        metavar="歌曲排序",
        default="default",
        help="【限制总量时】歌曲排序方式 (default: 默认排序 hot: 热度高（相对于其所在专辑）在前 time: 发行时间新在前)",
        choices=["default", "hot", "time"],
    )
    group.add_argument(
        "--reverse-sort",
        action="store_true",
        default=False,
        help="【限制总量时】倒序排序歌曲",
    )
    group.add_argument(
        "--user-bookmarks",
        action="store_true",
        default=False,
        help="【下载用户歌单时】在下载用户创建的歌单的同时，也下载其收藏的歌单",
    )
    group = parser.add_argument_group("工具")
    group.add_argument(
        "--save-m3u",
        metavar="保存M3U播放列表文件名",
        default="",
        help=r"""将本次下载的歌曲文件名依一定顺序保存在M3U文件中；写入的文件目录相对于该M3U文件
        文件编码为 UTF-8
        顺序为：链接先后优先——每个链接的所有歌曲依照歌曲排序设定 （--sort-by）排序""",
    )
    args = parser.parse_args()
    # Clean up
    args.lyric_no = args.lyric_no.lower()
    args.lyric_no = args.lyric_no.split(" ")
    if "none" in args.lyric_no:
        args.lyric_no = []

    def print_help_and_exit():
        sys.argv.append("-h")  # If using placeholder, no argument is really passed
        sys.exit(__main__())  # In which case, print help and exit

    if args.url == PLACEHOLDER_URL and not args.save:
        if quit_on_empty_args:
            print_help_and_exit()
        else:
            return args, []
    try:
        return args, [parse_sharelink(url) for url in args.url]
    except AssertionError:
        if args.url == PLACEHOLDER_URL:
            print_help_and_exit()
        assert args.save, "无效分享链接 %s" % " ".join(
            args.url
        )  # Allow invalid links for this one
        return args, []


def __main__(return_tasks=False):
    args, tasks = parse_args()
    log_stream = sys.stdout
    # Getting tqdm & logger to work nicely together
    if OPTIONALS["tqdm"]:
        from tqdm.std import tqdm as tqdm_c

        class SemaphoreStdout:
            @staticmethod
            def write(__s):
                # Blocks tqdm's output until write on this stream is done
                # Solves cases where progress bars gets re-rendered when logs
                # spews out too fast
                with tqdm_c.external_write_mode(file=sys.stdout, nolock=False):
                    return sys.stdout.write(__s)

        log_stream = SemaphoreStdout
    if args.log_level == "NOTSET":
        args.log_level = os.environ.get("PYNCM_DEBUG", "INFO")
    if OPTIONALS["coloredlogs"]:
        import coloredlogs

        coloredlogs.install(
            level=args.log_level,
            fmt="%(asctime)s %(name)s [%(levelname).4s] %(message)s",
            stream=log_stream,
            isatty=True,
        )
    basicConfig(
        level=args.log_level,
        format="[%(levelname).4s] %(name)s %(message)s",
        stream=log_stream,
    )
    from pyncm.utils.constant import known_good_deviceIds
    from random import choice as rnd_choice

    GetCurrentSession().deviceId = rnd_choice(known_good_deviceIds)
    # Pick a random one that WILL work!
    if args.deviceId:
        GetCurrentSession().deviceId = args.deviceId
    if args.load:
        logger.info("读取登录信息 : %s" % args.load)
        SetCurrentSession(LoadSessionFromString(open(args.load).read()))
    if args.http:
        GetCurrentSession().force_http = True
        logger.warning("优先使用 HTTP")
    if args.phone and args.pwd:
        login.LoginViaCellphone(args.phone, args.pwd)
    if args.cookie:
        login.LoginViaCookie(args.cookie)
    if not GetCurrentSession().logged_in:
        login.LoginViaAnonymousAccount()
        logger.info(
            "以匿名身份登陆成功，deviceId=%s, UID: %s"
            % (GetCurrentSession().deviceId, GetCurrentSession().uid)
        )
    executor = TaskPoolExecutorThread(max_workers=args.max_workers)
    if not GetCurrentSession().is_anonymous:
        logger.info(
            "账号 ：%s (VIP %s)"
            % (GetCurrentSession().nickname, GetCurrentSession().vipType)
        )
    if args.save:
        logger.info("保存登陆信息于 : %s" % args.save)
        open(args.save, "w").write(DumpSessionAsString(GetCurrentSession()))
        return 0
    executor.daemon = True
    executor.start()

    def enqueue_task(task):
        executor.task_queue.put(task)

    subroutines = []
    queuedTasks = []
    for rtype, ids in tasks:
        task_desc = "ID: %s 类型: %s" % (
            ids[0],
            {
                "album": "专辑",
                "playlist": "歌单®",
                "song": "单曲",
                "artist": "艺术家",
                "user": "用户",
            }[rtype],
        )
        logger.info("处理任务 %s" % task_desc)
        subroutine = create_subroutine(rtype)(args, enqueue_task)
        queuedTasks += subroutine(ids)  # Enqueue tasks
        subroutines.append((subroutine, task_desc))

    if OPTIONALS["tqdm"]:
        import tqdm

        _tqdm = tqdm.tqdm(
            bar_format="{desc}: {percentage:.1f}%|{bar}| {n:.2f}/{total_fmt} {elapsed}<{remaining}"
        )
        _tqdm.total = len(queuedTasks)

        def report():
            _tqdm.desc = _tqdm.format_sizeof(executor.xfered, suffix="B", divisor=1024)
            _tqdm.update(min(executor.finished_tasks, len(queuedTasks)) - _tqdm.n)
            return True

    else:

        def report():
            sys.stderr.write(
                f"下载中 : {executor.finished_tasks:.1f} / {len(queuedTasks)} ({(executor.finished_tasks * 100 / len(queuedTasks)):.1f} %,{executor.xfered >> 20} MB)               \r"
            )
            return True

    while executor.task_queue.unfinished_tasks >= 0:
        try:
            report() and sleep(0.5)
            if executor.task_queue.unfinished_tasks == 0:
                break
        except KeyboardInterrupt:
            break

    # Check final results
    # Thought: Maybe we should automatically retry these afterwards?
    failed_ids = dict()
    for routine, desc in subroutines:
        routine: Subroutine
        if routine.has_exceptions:
            logger.error("%s - 下载未完成" % desc)
            for exception_id, exceptions in routine.exceptions.items():
                failed_ids[exception_id] = True
                for exception in exceptions:
                    exception_obj, desc = exception
                    logger.warning(
                        "下载出错 ID: %s - %s%s"
                        % (exception_id, exception_obj, " (%s)" % desc if desc else "")
                    )

    if args.save_m3u:
        output_name = args.save_m3u
        output_folder = os.path.dirname(output_name)

        with open(output_name, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            for task in queuedTasks:
                task: TrackDownloadTask
                filePath = task.save_as + "." + task.extension
                relPath = os.path.relpath(filePath, output_folder)
                f.write("#EXTINF:,\n")
                f.write(relPath)
                f.write("\n")
        logger.info("已保存播放列表至：%s" % output_name)

    if failed_ids:
        logger.error("你可以将下载失败的 ID 作为参数以再次下载")
        logger.error(
            "所有失败的任务 ID: %s" % " ".join([str(i) for i in failed_ids.keys()])
        )
    report()
    logger.info(
        f"任务完成率 {(executor.finished_tasks * 100 / max(1,len(queuedTasks))):.1f}%"
    )
    # To get actually downloaded tasks, filter by exlcuding failed_ids against task.song.ID
    if return_tasks:
        return queuedTasks, failed_ids
    return


class PyNCMGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("网易云音乐下载器")
        self.setMinimumSize(1200, 700)  # 减小窗口默认尺寸
        self.setWindowIcon(QIcon("icon.ico"))
        
        # 配置文件路径
        self.config_file = Path.home() / '.pyncm' / 'config.json'
        self.config_file.parent.mkdir(exist_ok=True)
        
        # 加载配置
        self.load_config()
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建水平布局作为主布局
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(10)  # 减小间距
        main_layout.setContentsMargins(10, 10, 10, 10)  # 减小边距
        
        # 创建左侧控制面板
        left_layout = QVBoxLayout()
        self.create_login_section(left_layout)
        self.create_url_section(left_layout)
        self.create_options_section(left_layout)
        self.create_progress_section(left_layout)
        main_layout.addLayout(left_layout, 1)  # 设置拉伸因子为1
        
        # 创建右侧下载列表区域
        right_layout = QVBoxLayout()
        self.create_download_list_section(right_layout)
        main_layout.addLayout(right_layout, 2)  # 设置拉伸因子为2
        
        # 初始化下载管理器
        self.download_manager = None
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建状态栏
        self.statusBar().showMessage("就绪")

    def load_config(self):
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.last_output_dir = config.get('last_output_dir', '.')
                    # 加载登录信息
                    self.login_info = {
                        'type': config.get('login_type', 'anonymous'),
                        'phone': config.get('phone', ''),
                        'cookie': config.get('cookie', ''),
                        'last_login': config.get('last_login', '')
                    }
            else:
                self.last_output_dir = '.'
                self.login_info = {
                    'type': 'anonymous',
                    'phone': '',
                    'cookie': '',
                    'last_login': ''
                }
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
            self.last_output_dir = '.'
            self.login_info = {
                'type': 'anonymous',
                'phone': '',
                'cookie': '',
                'last_login': ''
            }

    def save_config(self):
        """保存配置文件"""
        try:
            config = {
                'last_output_dir': self.output_dir.text(),
                'login_type': self.login_info['type'],
                'phone': self.login_info['phone'],
                'cookie': self.login_info['cookie'],
                'last_login': self.login_info['last_login']
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存配置文件失败: {e}")

    def create_login_section(self, parent_layout):
        login_group = QGroupBox("登录 (可选)")
        login_layout = QVBoxLayout()
        
        # 添加登录方式选择
        login_type_layout = QHBoxLayout()
        self.login_type_label = QLabel("登录方式:")
        self.login_type_combo = QComboBox()
        self.login_type_combo.addItems(["匿名登录", "手机号登录", "Cookie登录", "扫码登录"])
        self.login_type_combo.currentIndexChanged.connect(self.on_login_type_changed)
        login_type_layout.addWidget(self.login_type_label)
        login_type_layout.addWidget(self.login_type_combo)
        login_layout.addLayout(login_type_layout)
        
        # 手机号登录部分
        self.phone_login_widget = QWidget()
        phone_layout = QHBoxLayout()
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("手机号")
        self.pwd_input = QLineEdit()
        self.pwd_input.setPlaceholderText("密码")
        self.pwd_input.setEchoMode(QLineEdit.EchoMode.Password)
        phone_layout.addWidget(QLabel("手机号:"))
        phone_layout.addWidget(self.phone_input)
        phone_layout.addWidget(QLabel("密码:"))
        phone_layout.addWidget(self.pwd_input)
        self.phone_login_widget.setLayout(phone_layout)
        
        # Cookie登录部分
        self.cookie_login_widget = QWidget()
        cookie_layout = QHBoxLayout()
        self.cookie_input = QLineEdit()
        self.cookie_input.setPlaceholderText("MUSIC_U Cookie")
        cookie_layout.addWidget(QLabel("Cookie:"))
        cookie_layout.addWidget(self.cookie_input)
        self.cookie_login_widget.setLayout(cookie_layout)
        
        # 登录按钮
        login_btn = QPushButton("登录")
        login_btn.clicked.connect(self.handle_login)
        
        # 添加登录信息显示
        self.login_info_label = QLabel("未登录")
        self.login_info_label.setStyleSheet("color: #666; font-size: 12px;")
        self.login_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 添加所有组件到主布局
        login_layout.addWidget(self.phone_login_widget)
        login_layout.addWidget(self.cookie_login_widget)
        login_layout.addWidget(login_btn)
        login_layout.addWidget(self.login_info_label)
        
        # 初始显示状态
        self.phone_login_widget.hide()
        self.cookie_login_widget.hide()
        
        login_group.setLayout(login_layout)
        parent_layout.addWidget(login_group)
        
        # 尝试自动登录
        self.auto_login()

    def on_login_type_changed(self, index):
        # 隐藏所有登录方式
        self.phone_login_widget.hide()
        self.cookie_login_widget.hide()
        
        # 显示选中的登录方式
        if index == 1:  # 手机号登录
            self.phone_login_widget.show()
        elif index == 2:  # Cookie登录
            self.cookie_login_widget.show()
        elif index == 3:  # 扫码登录
            self.show_qr_login()

    def show_qr_login(self):
        """显示扫码登录窗口"""
        qr_window = QRLoginWindow(self)
        if qr_window.exec() == QDialog.DialogCode.Accepted and qr_window.music_u_cookie:
            try:
                # 使用获取到的cookie进行登录
                login.LoginViaCookie(qr_window.music_u_cookie)
                self.login_info = {
                    'type': 'cookie',
                    'phone': '',
                    'cookie': qr_window.music_u_cookie,
                    'last_login': ''
                }
                # 保存登录信息
                self.save_config()
                # 更新登录状态
                self.update_login_status()
            except Exception as e:
                QMessageBox.critical(self, "登录失败", str(e))

    def auto_login(self):
        """尝试自动登录"""
        try:
            if self.login_info['type'] == 'phone' and self.login_info['phone']:
                login.LoginViaCellphone(self.login_info['phone'], self.login_info['cookie'])
                self.update_login_status()
            elif self.login_info['type'] == 'cookie' and self.login_info['cookie']:
                login.LoginViaCookie(self.login_info['cookie'])
                self.update_login_status()
            else:
                login.LoginViaAnonymousAccount()
                self.update_login_status()
        except Exception as e:
            logger.warning(f"自动登录失败: {e}")
            # 如果自动登录失败，尝试匿名登录
            try:
                login.LoginViaAnonymousAccount()
                self.update_login_status()
            except:
                pass

    def handle_login(self):
        login_type = self.login_type_combo.currentIndex()
        
        try:
            if login_type == 0:  # 匿名登录
                login.LoginViaAnonymousAccount()
                self.login_info = {
                    'type': 'anonymous',
                    'phone': '',
                    'cookie': '',
                    'last_login': ''
                }
            elif login_type == 1:  # 手机号登录
                phone = self.phone_input.text()
                pwd = self.pwd_input.text()
                if not phone or not pwd:
                    QMessageBox.warning(self, "警告", "请输入手机号和密码")
                    return
                login.LoginViaCellphone(phone, pwd)
                self.login_info = {
                    'type': 'phone',
                    'phone': phone,
                    'cookie': pwd,  # 注意：这里存储密码可能不安全，实际应用中应该使用更安全的方式
                    'last_login': ''
                }
            elif login_type == 2:  # Cookie登录
                cookie = self.cookie_input.text()
                if not cookie:
                    QMessageBox.warning(self, "警告", "请输入Cookie")
                    return
                login.LoginViaCookie(cookie)
                self.login_info = {
                    'type': 'cookie',
                    'phone': '',
                    'cookie': cookie,
                    'last_login': ''
                }
            
            # 保存登录信息
            self.save_config()
            
            # 登录成功后更新UI状态
            self.update_login_status()
            
        except Exception as e:
            QMessageBox.critical(self, "登录失败", str(e))

    def update_login_status(self):
        session = GetCurrentSession()
        if session.logged_in:
            if session.is_anonymous:
                status_text = f"匿名登录成功 (UID: {session.uid})"
            else:
                status_text = f"登录成功 - {session.nickname} (VIP {session.vipType})"
            self.login_info_label.setText(status_text)
            self.login_info_label.setStyleSheet("color: #4CAF50; font-size: 12px;")
        else:
            self.login_info_label.setText("未登录")
            self.login_info_label.setStyleSheet("color: #666; font-size: 12px;")

    def create_url_section(self, parent_layout):
        url_group = QGroupBox("下载链接")
        url_layout = QVBoxLayout()
        
        # 添加链接输入框
        self.url_input = QTextEdit()
        self.url_input.setPlaceholderText("输入网易云音乐分享链接，每行一个")
        url_layout.addWidget(self.url_input)
        
        # 添加下载按钮
        download_btn = QPushButton("开始下载")
        download_btn.clicked.connect(self.start_download)
        url_layout.addWidget(download_btn)
        
        url_group.setLayout(url_layout)
        parent_layout.addWidget(url_group)
        
    def create_options_section(self, parent_layout):
        options_group = QGroupBox("下载选项")
        options_layout = QFormLayout()
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["lossless", "standard", "exhigh", "hires"])
        self.quality_combo.setCurrentText("lossless")
        
        self.output_dir = QLineEdit()
        self.output_dir.setText(self.last_output_dir)  # 使用保存的目录
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_output_dir)
        
        options_layout.addRow("音质:", self.quality_combo)
        options_layout.addRow("保存位置:", self.output_dir)
        options_layout.addRow("", self.browse_btn)
        
        options_group.setLayout(options_layout)
        parent_layout.addWidget(options_group)
        
    def create_progress_section(self, parent_layout):
        progress_group = QGroupBox("下载进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("就绪")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        parent_layout.addWidget(progress_group)
    
    def handle_cell_click(self, row, column):
        """处理表格单元格点击事件"""
        if column == 0:  # 只处理第一列（复选框列）的点击
            # 获取所有选中的行
            selected_rows = set(item.row() for item in self.playlist_table.selectedItems())
            
            # 如果没有其他选中的行，就只切换当前行的状态
            if not selected_rows or (len(selected_rows) == 1 and row in selected_rows):
                item = self.playlist_table.item(row, 0)
                if item:
                    current_state = item.checkState()
                    new_state = Qt.CheckState.Checked if current_state == Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
                    item.setCheckState(new_state)
                    # 更新选中计数
                    self.update_selection_counter()
            # 如果有其他选中的行，则切换所有选中行的状态
            else:
                item = self.playlist_table.item(row, 0)
                if item:
                    new_state = Qt.CheckState.Checked if item.checkState() == Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
                    for selected_row in selected_rows:
                        selected_item = self.playlist_table.item(selected_row, 0)
                        if selected_item:
                            selected_item.setCheckState(new_state)
                    # 更新选中计数
                    self.update_selection_counter()
    
    def handle_cell_changed(self, row, column):
        """处理表格单元格变化事件"""
        if column == 0:  # 只处理第一列（复选框列）的变化
            item = self.playlist_table.item(row, column)
            if item:
                # 确保复选框状态正确设置
                if not item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                # 更新选中计数
                self.update_selection_counter()

    def handle_selection_changed(self):
        """处理表格选择变化事件"""
        # 获取所有选中的行
        selected_rows = set(item.row() for item in self.playlist_table.selectedItems())
        
        # 如果有选中的行，更新它们的复选框状态
        if selected_rows:
            # 获取第一个选中项的状态，用于决定是全选还是全不选
            first_row = min(selected_rows)
            first_item = self.playlist_table.item(first_row, 0)
            if first_item:
                new_state = Qt.CheckState.Checked if first_item.checkState() == Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
                
                # 更新所有选中行的复选框状态
                for row in selected_rows:
                    item = self.playlist_table.item(row, 0)
                    if item:
                        item.setCheckState(new_state)
                # 更新选中计数
                self.update_selection_counter()

    def update_selection_counter(self):
        """更新选中歌曲计数器"""
        checked_count = 0
        total_count = self.playlist_table.rowCount()
        
        for row in range(total_count):
            item = self.playlist_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                checked_count += 1
        
        # 更新计数器显示
        self.selection_counter.setText(f"已选择: {checked_count} / {total_count} 首")
        
        # 更新全选按钮的文本
        if checked_count == total_count and total_count > 0:
            self.select_all_btn.setText("取消全选")
        else:
            self.select_all_btn.setText("全选")

    def toggle_select_all(self):
        """切换全选/取消全选状态"""
        is_select_all = self.select_all_btn.text() == "全选"
        
        # 更新所有复选框的状态
        for row in range(self.playlist_table.rowCount()):
            item = self.playlist_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked if is_select_all else Qt.CheckState.Unchecked)
        
        # 更新选中计数
        self.update_selection_counter()
        
        # 更新按钮文本 - 不需要在这里更新，因为update_selection_counter已经会处理

    def create_download_list_section(self, parent_layout):
        # 创建右侧主布局
        right_layout = QVBoxLayout()
        
        # 创建歌单区域
        playlist_group = QGroupBox("歌单")
        playlist_layout = QVBoxLayout()
        
        # 添加歌单URL输入
        url_layout = QHBoxLayout()
        self.playlist_url_input = QLineEdit()
        self.playlist_url_input.setPlaceholderText("输入歌单URL，例如：https://music.163.com/#/playlist?id=3559356")
        self.playlist_url_input.setText("https://music.163.com/#/playlist?id=3559356")  # 设置默认值
        load_btn = QPushButton("加载歌单")
        load_btn.clicked.connect(self.load_playlist)
        url_layout.addWidget(self.playlist_url_input)
        url_layout.addWidget(load_btn)
        playlist_layout.addLayout(url_layout)

        # 添加选中计数器
        self.selection_counter = QLabel("已选择: 0 首")
        self.selection_counter.setStyleSheet("""
            QLabel {
                color: #2196F3;
                font-size: 12px;
                padding: 5px;
                background: rgba(33, 150, 243, 0.1);
                border-radius: 4px;
                margin: 5px 0;
            }
        """)
        playlist_layout.addWidget(self.selection_counter)
        
        # 添加歌单歌曲列表
        self.playlist_table = QTableWidget()
        self.playlist_table.setColumnCount(5)
        self.playlist_table.setHorizontalHeaderLabels(["选择", "歌曲", "歌手", "专辑", "ID"])
        
        # 调整列宽比例
        self.playlist_table.setColumnWidth(0, 50)
        self.playlist_table.setColumnWidth(1, 200)
        self.playlist_table.setColumnWidth(2, 150)
        self.playlist_table.setColumnWidth(3, 150)
        self.playlist_table.setColumnWidth(4, 100)
        
        # 设置表格的其他属性
        self.playlist_table.setAlternatingRowColors(True)
        self.playlist_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)  # 允许多选
        self.playlist_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.playlist_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.playlist_table.horizontalHeader().setStretchLastSection(True)
        self.playlist_table.verticalHeader().setVisible(False)
        
        # 添加单元格点击事件处理
        self.playlist_table.cellClicked.connect(self.handle_cell_click)
        self.playlist_table.cellChanged.connect(self.handle_cell_changed)
        
        # 添加选择变化事件处理
        self.playlist_table.itemSelectionChanged.connect(self.handle_selection_changed)
        
        # 设置默认行高
        self.playlist_table.verticalHeader().setDefaultSectionSize(25)
        
        # 设置表头样式
        header = self.playlist_table.horizontalHeader()
        header.setFixedHeight(25)
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #323232;
                padding: 2px;
                border: 1px solid #3d3d3d;
                font-weight: bold;
                color: #ffffff;
                font-size: 12px;
            }
        """)
        
        # 设置表格样式
        self.playlist_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                background-color: #2b2b2b;
                gridline-color: #3d3d3d;
                color: #ffffff;
            }
            QTableWidget::item {
                padding: 2px;
                color: #ffffff;
                font-size: 12px;
                border-bottom: 1px solid #3d3d3d;
            }
            QTableWidget::item:selected {
                background-color: rgba(33, 150, 243, 0.2);
                color: #ffffff;
            }
            QTableWidget::item:alternate {
                background-color: #323232;
            }
            QTableWidget::item:alternate:selected {
                background-color: rgba(33, 150, 243, 0.2);
                color: #ffffff;
            }
            QTableWidget::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                margin: 3px;
            }
            QTableWidget::indicator:unchecked {
                background-color: transparent;
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
            QTableWidget::indicator:unchecked:hover {
                border-color: rgba(33, 150, 243, 0.5);
            }
            QTableWidget::indicator:checked {
                background-color: #2196F3;
                border: none;
                image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='18' height='18' viewBox='0 0 18 18'%3E%3Cpath d='M6.5 12.5l-4-4 1.5-1.5 2.5 2.5 5.5-5.5 1.5 1.5z' fill='white'/%3E%3C/svg%3E");
            }
            QTableWidget::item:selected:active {
                background-color: rgba(33, 150, 243, 0.3);
                color: #ffffff;
            }
            QTableWidget::item:selected:!active {
                background-color: rgba(33, 150, 243, 0.2);
                color: #ffffff;
            }
            QTableWidget::item:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
            QTableWidget::item:selected:hover {
                background-color: rgba(33, 150, 243, 0.4);
            }
        """)
        
        playlist_layout.addWidget(self.playlist_table)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 添加全选/取消全选按钮
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.select_all_btn.clicked.connect(self.toggle_select_all)
        button_layout.addWidget(self.select_all_btn)
        
        # 添加下载选中歌曲按钮
        download_selected_btn = QPushButton("下载选中歌曲")
        download_selected_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:pressed {
                background-color: #1B5E20;
            }
        """)
        download_selected_btn.clicked.connect(self.download_selected_songs)
        button_layout.addWidget(download_selected_btn)
        
        playlist_layout.addLayout(button_layout)
        
        playlist_group.setLayout(playlist_layout)
        right_layout.addWidget(playlist_group)
        
        # 创建下载列表区域
        list_group = QGroupBox("下载列表")
        list_layout = QVBoxLayout()
        
        self.download_list = QTableWidget()
        self.download_list.setColumnCount(3)
        self.download_list.setHorizontalHeaderLabels(["歌曲", "状态", "进度"])
        
        # 调整列宽比例
        self.download_list.setColumnWidth(0, 400)
        self.download_list.setColumnWidth(1, 100)
        self.download_list.setColumnWidth(2, 100)
        
        # 设置表格的其他属性
        self.download_list.setAlternatingRowColors(True)
        self.download_list.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.download_list.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.download_list.horizontalHeader().setStretchLastSection(True)
        self.download_list.verticalHeader().setVisible(False)
        
        # 设置默认行高
        self.download_list.verticalHeader().setDefaultSectionSize(25)
        
        # 设置表头样式
        header = self.download_list.horizontalHeader()
        header.setFixedHeight(25)
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #323232;
                padding: 2px;
                border: 1px solid #3d3d3d;
                font-weight: bold;
                color: #ffffff;
                font-size: 12px;
            }
        """)
        
        # 设置表格样式
        self.download_list.setStyleSheet("""
            QTableWidget {
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                background-color: #2b2b2b;
                gridline-color: #3d3d3d;
                color: #ffffff;
            }
            QTableWidget::item {
                padding: 2px;
                color: #ffffff;
                font-size: 12px;
            }
            QTableWidget::item:selected {
                background-color: #3d3d3d;
            }
            QTableWidget::item:alternate {
                background-color: #323232;
            }
        """)
        
        list_layout.addWidget(self.download_list)
        list_group.setLayout(list_layout)
        right_layout.addWidget(list_group)
        
        parent_layout.addLayout(right_layout)

    def handle_cell_click(self, row, column):
        """处理表格单元格点击事件"""
        if column == 0:  # 只处理第一列（复选框列）的点击
            # 获取所有选中的行
            selected_rows = set(item.row() for item in self.playlist_table.selectedItems())
            
            # 如果没有其他选中的行，就只切换当前行的状态
            if not selected_rows or (len(selected_rows) == 1 and row in selected_rows):
                item = self.playlist_table.item(row, 0)
                if item:
                    current_state = item.checkState()
                    new_state = Qt.CheckState.Checked if current_state == Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
                    item.setCheckState(new_state)
                    # 更新选中计数
                    self.update_selection_counter()
            # 如果有其他选中的行，则切换所有选中行的状态
            else:
                item = self.playlist_table.item(row, 0)
                if item:
                    new_state = Qt.CheckState.Checked if item.checkState() == Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
                    for selected_row in selected_rows:
                        selected_item = self.playlist_table.item(selected_row, 0)
                        if selected_item:
                            selected_item.setCheckState(new_state)
                    # 更新选中计数
                    self.update_selection_counter()

    def handle_cell_changed(self, row, column):
        """处理表格单元格变化事件"""
        if column == 0:  # 只处理第一列（复选框列）的变化
            item = self.playlist_table.item(row, column)
            if item:
                # 确保复选框状态正确设置
                if not item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                # 更新选中计数
                self.update_selection_counter()

    def handle_selection_changed(self):
        """处理表格选择变化事件"""
        # 获取所有选中的行
        selected_rows = set(item.row() for item in self.playlist_table.selectedItems())
        
        # 如果有选中的行，更新它们的复选框状态
        if selected_rows:
            # 获取第一个选中项的状态，用于决定是全选还是全不选
            first_row = min(selected_rows)
            first_item = self.playlist_table.item(first_row, 0)
            if first_item:
                new_state = Qt.CheckState.Checked if first_item.checkState() == Qt.CheckState.Unchecked else Qt.CheckState.Unchecked
                
                # 更新所有选中行的复选框状态
                for row in selected_rows:
                    item = self.playlist_table.item(row, 0)
                    if item:
                        item.setCheckState(new_state)
                # 更新选中计数
                self.update_selection_counter()

    def update_selection_counter(self):
        """更新选中歌曲计数器"""
        checked_count = 0
        total_count = self.playlist_table.rowCount()
        
        for row in range(total_count):
            item = self.playlist_table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                checked_count += 1
        
        # 更新计数器显示
        self.selection_counter.setText(f"已选择: {checked_count} / {total_count} 首")
        
        # 更新全选按钮的文本
        if checked_count == total_count and total_count > 0:
            self.select_all_btn.setText("取消全选")
        else:
            self.select_all_btn.setText("全选")

    def toggle_select_all(self):
        """切换全选/取消全选状态"""
        is_select_all = self.select_all_btn.text() == "全选"
        
        # 更新所有复选框的状态
        for row in range(self.playlist_table.rowCount()):
            item = self.playlist_table.item(row, 0)
            if item:
                item.setCheckState(Qt.CheckState.Checked if is_select_all else Qt.CheckState.Unchecked)
        
        # 更新选中计数
        self.update_selection_counter()
        
        # 更新按钮文本 - 不需要在这里更新，因为update_selection_counter已经会处理

    def load_playlist(self):
        """加载歌单内容"""
        url = self.playlist_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "警告", "请输入歌单URL")
            return
            
        # 检查登录状态
        session = GetCurrentSession()
        if not session.logged_in or session.is_anonymous:
            QMessageBox.warning(self, "警告", "请先登录网易云音乐账号")
            return
            
        try:
            # 解析歌单ID
            rtype, ids = parse_sharelink(url)
            if rtype != "playlist":
                QMessageBox.warning(self, "警告", "请输入正确的歌单URL")
                return
                
            # 获取歌单信息
            playlist_info = playlist.GetPlaylistInfo(ids[0])
            if not playlist_info or "playlist" not in playlist_info:
                QMessageBox.warning(self, "警告", "获取歌单信息失败")
                return
                
            # 清空现有列表
            self.playlist_table.setRowCount(0)
            
            # 获取歌单中的歌曲
            track_ids = [tid.get("id") for tid in playlist_info["playlist"]["trackIds"]]
            songs_info = track.GetTrackDetail(track_ids).get("songs", [])
            
            # 添加歌曲到表格
            for song in songs_info:
                row = self.playlist_table.rowCount()
                self.playlist_table.insertRow(row)
                
                # 添加复选框
                checkbox = QTableWidgetItem()
                checkbox.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                checkbox.setCheckState(Qt.CheckState.Unchecked)
                self.playlist_table.setItem(row, 0, checkbox)
                
                # 添加歌曲信息
                self.playlist_table.setItem(row, 1, QTableWidgetItem(song.get("name", "")))
                
                # 安全地处理艺术家信息
                artists = []
                for ar in song.get("ar", []):
                    if ar and isinstance(ar, dict):
                        name = ar.get("name")
                        if name:
                            artists.append(name)
                self.playlist_table.setItem(row, 2, QTableWidgetItem(", ".join(artists)))
                
                # 安全地处理专辑信息
                album_name = ""
                if isinstance(song.get("al"), dict):
                    album_name = song["al"].get("name", "")
                self.playlist_table.setItem(row, 3, QTableWidgetItem(album_name))
                
                # 存储歌曲ID
                song_id = str(song.get("id", ""))
                id_item = QTableWidgetItem(song_id)
                self.playlist_table.setItem(row, 4, id_item)
                
            # 设置表格列宽
            self.playlist_table.setColumnWidth(0, 50)  # 选择列
            self.playlist_table.setColumnWidth(1, 200)  # 歌曲列
            self.playlist_table.setColumnWidth(2, 150)  # 歌手列
            self.playlist_table.setColumnWidth(3, 150)  # 专辑列
            self.playlist_table.setColumnWidth(4, 100)  # ID列
            
            # 初始化选中计数器
            self.update_selection_counter()
                
            QMessageBox.information(self, "成功", f"已加载歌单：{playlist_info['playlist']['name']}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载歌单失败：{str(e)}")
            logger.exception("加载歌单失败")

    def download_selected_songs(self):
        """下载选中的歌曲"""
        selected_urls = []
        # 获取所有选中的歌曲
        for row in range(self.playlist_table.rowCount()):
            checkbox_item = self.playlist_table.item(row, 0)
            if checkbox_item and checkbox_item.checkState() == Qt.CheckState.Checked:
                id_item = self.playlist_table.item(row, 4)
                if id_item and id_item.text():
                    song_id = id_item.text()
                    song_url = f"https://music.163.com/#/song?id={song_id}"
                    selected_urls.append(song_url)
        if not selected_urls:
            QMessageBox.warning(self, "警告", "请选择要下载的歌曲")
            return
        # 将选中的歌曲URL添加到下载链接输入框
        current_urls = self.url_input.toPlainText().strip()
        if current_urls:
            current_urls += "\n"
        self.url_input.setText(current_urls + "\n".join(selected_urls))
        # 自动开始下载
        self.start_download()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        self.status_label.setText(message)
        
    def download_finished(self):
        QMessageBox.information(self, "完成", "下载完成")

    def update_download_list(self, title, status):
        row = self.download_list.rowCount()
        self.download_list.insertRow(row)
        
        # 创建并设置歌曲名单元格
        title_item = QTableWidgetItem(title)
        title_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.download_list.setItem(row, 0, title_item)
        
        # 创建并设置状态单元格
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.download_list.setItem(row, 1, status_item)
        
        # 创建并设置进度单元格
        progress_item = QTableWidgetItem("0%")
        progress_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.download_list.setItem(row, 2, progress_item)
        
        # 设置行高
        self.download_list.setRowHeight(row, 25)  # 减少行高

    def update_task_status(self, row, status):
        status_item = QTableWidgetItem(status)
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.download_list.setItem(row, 1, status_item)

    def update_task_progress(self, row, progress):
        progress_item = QTableWidgetItem(f"{progress}%")
        progress_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.download_list.setItem(row, 2, progress_item)

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_about(self):
        QMessageBox.about(self, "关于", 
            "网易云音乐下载器 v1.0\n\n"
            "一个简单的网易云音乐下载工具\n"
            "支持下载歌曲、歌单、专辑等")

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.output_dir.text())
        if dir_path:
            self.output_dir.setText(dir_path)
            self.save_config()  # 保存新的目录选择

    def start_download(self):
        urls = self.url_input.toPlainText().strip().split('\n')
        if not urls:
            QMessageBox.warning(self, "警告", "请输入下载链接")
            return
            
        # 清空下载列表
        self.download_list.setRowCount(0)
        
        # 创建一个模拟的 args 对象，包含所有必要的属性
        class Args:
            def __init__(self, quality, output):
                self.quality = quality
                self.output = output
                self.max_workers = 4
                self.output_name = "{title}"
                self.lyric_no = ["yrc"]
                self.no_overwrite = False
                self.count = 0
                self.sort_by = "default"
                self.reverse_sort = False
                self.use_download_api = False
                self.save_m3u = ""
                self.user_bookmarks = False
                self.http = False
                self.deviceId = ""
        
        # 从GUI组件获取值并创建Args对象
        options = Args(
            quality=self.quality_combo.currentText(),
            output=self.output_dir.text()
        )
        
        self.download_worker = DownloadWorker(urls, options)
        self.download_worker.progress_updated.connect(self.update_progress)
        self.download_worker.status_updated.connect(self.update_status)
        self.download_worker.download_completed.connect(self.download_finished)
        self.download_worker.task_added.connect(self.update_download_list)
        self.download_worker.task_progress_updated.connect(self.update_task_progress)
        self.download_worker.task_status_updated.connect(self.update_task_status)
        self.download_worker.start()


class DownloadWorker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    download_completed = pyqtSignal()
    task_added = pyqtSignal(str, str)
    task_progress_updated = pyqtSignal(int, int)
    task_status_updated = pyqtSignal(int, str)
    
    def __init__(self, urls, options, is_song_list=False):
        super().__init__()
        self.urls = urls
        self.options = options
        self.executor = TaskPoolExecutorThread(max_workers=options.max_workers)
        self.executor.start()
        self.task_rows = {}
        self.current_row = 0
        self.is_song_list = is_song_list
        self.task_progress = {}  # 存储每个任务的进度
        
    def run(self):
        try:
            queued_tasks = []
            
            if self.is_song_list:
                # 直接处理歌曲ID列表
                try:
                    # 获取歌曲详情
                    songs_info = track.GetTrackDetail(self.urls).get("songs", [])
                    if not songs_info:
                        self.status_updated.emit("未找到可下载的歌曲")
                        self.download_completed.emit()
                        return
                        
                    # 创建下载任务
                    subroutine = create_subroutine("song")(self.options, self.executor.task_queue.put)
                    tasks = subroutine([song["id"] for song in songs_info])
                    
                    if tasks:
                        queued_tasks.extend(tasks)
                        
                        # 更新下载列表
                        for task in tasks:
                            if hasattr(task, 'song') and hasattr(task.song, 'Title'):
                                self.task_added.emit(
                                    task.song.Title,
                                    "等待中"
                                )
                                self.task_rows[task.song.ID] = self.current_row
                                self.task_progress[task.song.ID] = 0  # 初始化进度
                                self.current_row += 1
                            else:
                                self.task_added.emit(
                                    f"未知歌曲 (ID: {task.id if hasattr(task, 'id') else 'unknown'})",
                                    "等待中"
                                )
                                self.current_row += 1
                except Exception as e:
                    self.status_updated.emit(f"处理歌曲列表失败: {str(e)}")
                    logger.exception("处理歌曲列表失败")
                    return
            else:
                # 处理URL列表
                for url in self.urls:
                    try:
                        # 解析URL
                        rtype, ids = parse_sharelink(url)
                        self.status_updated.emit(f"正在处理: {url}")
                        
                        # 创建下载任务
                        subroutine = create_subroutine(rtype)(self.options, self.executor.task_queue.put)
                        tasks = subroutine(ids)
                        
                        if tasks:
                            queued_tasks.extend(tasks)
                            
                            # 更新下载列表
                            for task in tasks:
                                if hasattr(task, 'song') and hasattr(task.song, 'Title'):
                                    self.task_added.emit(
                                        task.song.Title,
                                        "等待中"
                                    )
                                    self.task_rows[task.song.ID] = self.current_row
                                    self.task_progress[task.song.ID] = 0  # 初始化进度
                                    self.current_row += 1
                                else:
                                    self.task_added.emit(
                                        f"未知歌曲 (ID: {task.id if hasattr(task, 'id') else 'unknown'})",
                                        "等待中"
                                    )
                                    self.current_row += 1
                        else:
                            self.status_updated.emit(f"未找到可下载的内容: {url}")
                            
                    except Exception as e:
                        self.status_updated.emit(f"处理链接失败: {url} - {str(e)}")
                        logger.exception(f"处理链接失败: {url}")
                        continue
            
            if not queued_tasks:
                self.status_updated.emit("没有可下载的任务")
                self.download_completed.emit()
                return
                
            # 等待所有任务完成
            total_tasks = len(queued_tasks)
            finished_tasks = 0
            
            while self.executor.task_queue.unfinished_tasks > 0:
                # 计算总体进度
                finished_tasks = total_tasks - self.executor.task_queue.unfinished_tasks
                total_progress = int((finished_tasks / total_tasks) * 100)
                
                # 更新总体进度
                self.progress_updated.emit(total_progress)
                self.status_updated.emit(
                    f"已下载: {self.executor.xfered >> 20} MB"
                )
                
                # 更新每个任务的进度
                for task in queued_tasks:
                    if hasattr(task, 'song'):
                        task_id = task.song.ID
                        if task_id in self.task_rows:
                            row = self.task_rows[task_id]
                            
                            # 计算单个任务的进度
                            if task_id in self.task_progress:
                                # 根据总体进度和任务完成情况更新单个任务进度
                                task_progress = min(100, int((finished_tasks / total_tasks) * 100))
                                if task_progress > self.task_progress[task_id]:
                                    self.task_progress[task_id] = task_progress
                                    self.task_progress_updated.emit(row, task_progress)
                                    
                            # 更新状态
                            if finished_tasks == total_tasks:
                                self.task_status_updated.emit(row, "已完成")
                            else:
                                self.task_status_updated.emit(row, "下载中")
                
                sleep(0.1)  # 更频繁地更新进度
                
            # 确保所有任务显示100%进度
            for task in queued_tasks:
                if hasattr(task, 'song'):
                    task_id = task.song.ID
                    if task_id in self.task_rows:
                        row = self.task_rows[task_id]
                        self.task_status_updated.emit(row, "已完成")
                        self.task_progress_updated.emit(row, 100)
            
            # 确保最终进度显示为100%
            self.progress_updated.emit(100)
            self.download_completed.emit()
            
        except Exception as e:
            self.status_updated.emit(f"下载出错: {str(e)}")
            logger.exception("下载过程出错")


class QRLoginWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("扫码登录")
        self.setMinimumSize(800, 600)  # 增加窗口宽度
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)  # 添加最大化按钮
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距
        
        # 创建网页视图
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl("https://music.163.com/#/login"))
        
        # 设置cookie监听
        self.profile = QWebEngineProfile.defaultProfile()
        self.profile.cookieStore().loadAllCookies()
        self.profile.cookieStore().cookieAdded.connect(self.on_cookie_added)
        
        # 添加网页视图到布局
        layout.addWidget(self.web_view)
        
        # 存储找到的MUSIC_U cookie
        self.music_u_cookie = None
        
    def on_cookie_added(self, cookie):
        """监听cookie变化"""
        if cookie.name() == b'MUSIC_U':
            self.music_u_cookie = cookie.value().data().decode()
            # 找到MUSIC_U cookie后关闭窗口
            self.accept()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = PyNCMGUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"启动失败: {str(e)}")
        sys.exit(1)
