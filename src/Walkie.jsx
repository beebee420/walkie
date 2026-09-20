import { useState, useRef, useEffect, useCallback } from "react";
import { Play, Pause, Mic, X, SkipForward, SkipBack, Radio, User, ArrowLeft, Volume2, Maximize2, RotateCcw, RotateCw, Plus, Camera } from "lucide-react";
import { supabase } from "./lib/supabaseClient";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";

const MAX_CAPTION_LENGTH = 120;

const ME = "you"; // internal sentinel meaning "the signed-in user" — see displayName()

const MAX_RECORD_SECONDS = 120;

// how long ago a post/reply was created, in the app's short relative style
function timeAgo(isoString) {
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(isoString).getTime()) / 1000));
  if (seconds < 60) return "now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  const days = Math.floor(hours / 24);
  return `${days}d`;
}

function getPublicAudioUrl(path) {
  if (!path) return null;
  return supabase.storage.from("audio").getPublicUrl(path).data.publicUrl;
}

function writeWavString(view, offset, str) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

function interleaveChannels(left, right) {
  const length = left.length + right.length;
  const result = new Float32Array(length);
  let index = 0;
  let inputIndex = 0;
  while (index < length) {
    result[index++] = left[inputIndex];
    result[index++] = right[inputIndex];
    inputIndex++;
  }
  return result;
}

// converts a decoded AudioBuffer (e.g. from a video's audio track) into a
// real, standalone audio/wav Blob that can be uploaded and played normally
function audioBufferToWavBlob(audioBuffer) {
  const numChannels = Math.min(2, audioBuffer.numberOfChannels);
  const sampleRate = audioBuffer.sampleRate;
  const bitDepth = 16;

  const samples =
    numChannels === 2
      ? interleaveChannels(audioBuffer.getChannelData(0), audioBuffer.getChannelData(1))
      : audioBuffer.getChannelData(0);

  const dataLength = samples.length * (bitDepth / 8);
  const buffer = new ArrayBuffer(44 + dataLength);
  const view = new DataView(buffer);

  writeWavString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLength, true);
  writeWavString(view, 8, "WAVE");
  writeWavString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
  view.setUint16(32, numChannels * (bitDepth / 8), true);
  view.setUint16(34, bitDepth, true);
  writeWavString(view, 36, "data");
  view.setUint32(40, dataLength, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: "audio/wav" });
}

let ffmpegLoadPromise = null;

// loads (once, cached for the session) a WebAssembly build of FFmpeg that
// runs entirely in the browser — used to pull audio out of a video directly,
// without needing to play through it in real time
function getFFmpeg() {
  if (!ffmpegLoadPromise) {
    ffmpegLoadPromise = (async () => {
      const ffmpeg = new FFmpeg();
      const baseURL = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/umd";
      await ffmpeg.load({
        coreURL: `${baseURL}/ffmpeg-core.js`,
        wasmURL: `${baseURL}/ffmpeg-core.wasm`,
      });
      return ffmpeg;
    })();
  }
  return ffmpegLoadPromise;
}

// fast path: process the video file directly via ffmpeg.wasm instead of
// playing through it. Tries a lossless stream copy first (near-instant),
// and only re-encodes if the source audio codec can't go straight into m4a.
async function extractAudioFromVideoViaFFmpeg(file) {
  const ffmpeg = await getFFmpeg();
  const ext = file.name.match(/\.[a-zA-Z0-9]+$/)?.[0] || ".mov";
  const inputName = `input${ext}`;

  await ffmpeg.writeFile(inputName, await fetchFile(file));

  const cleanup = async (...names) => {
    for (const n of names) {
      await ffmpeg.deleteFile(n).catch(() => {});
    }
  };

  try {
    await ffmpeg.exec(["-i", inputName, "-vn", "-acodec", "copy", "output.m4a"]);
    const data = await ffmpeg.readFile("output.m4a");
    await cleanup(inputName, "output.m4a");
    return new Blob([data.buffer], { type: "audio/mp4" });
  } catch (copyErr) {
    console.warn("ffmpeg stream copy failed, re-encoding instead:", copyErr);
    await ffmpeg.exec(["-i", inputName, "-vn", "-acodec", "libmp3lame", "-b:a", "128k", "output.mp3"]);
    const data = await ffmpeg.readFile("output.mp3");
    await cleanup(inputName, "output.mp3");
    return new Blob([data.buffer], { type: "audio/mpeg" });
  }
}

// extracts just the audio track from a video file, returning a real
// audio-only Blob — lets someone upload a video and post its audio.
// Only tries fast, non-realtime methods; if both fail, the caller shows an
// immediate error rather than falling back to a slow real-time approach.
async function extractAudioFromVideo(file) {
  try {
    return await extractAudioFromVideoViaFFmpeg(file);
  } catch (ffmpegErr) {
    console.warn("ffmpeg extraction failed, falling back to decodeAudioData:", ffmpegErr);
  }
  const arrayBuffer = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioCtx();
  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    return audioBufferToWavBlob(audioBuffer);
  } finally {
    audioCtx.close();
  }
}

// uploads a browser blob: URL's underlying audio data to the 'audio' bucket,
// returning the storage path (not the public URL) for saving on the row
async function uploadAudioToStorage(blobUrl, userId) {
  const blob = await (await fetch(blobUrl)).blob();
  const ext = blob.type.includes("webm")
    ? "webm"
    : blob.type.includes("wav")
    ? "wav"
    : blob.type.includes("mp4")
    ? "m4a"
    : "mp3";
  const path = `${userId}/${Date.now()}-${Math.random().toString(36).slice(2)}.${ext}`;
  const { error } = await supabase.storage.from("audio").upload(path, blob, {
    contentType: blob.type || "audio/webm",
  });
  if (error) throw error;
  return path;
}

function fmt(sec) {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r.toString().padStart(2, "0")}`;
}

// Some browsers report a blob-URL <audio> element's duration as Infinity
// until it's been forced to compute the real value (seek near the end, then
// back). This resolves once the real duration is known. While priming is in
// progress the element's currentTime jumps around internally — callers that
// listen for timeupdate must check audio.__priming and ignore those events,
// or they'll mistake the jump for having reached the end of the clip.
// Decodes a real waveform (peak amplitude per bar) from an audio blob URL so
// the trim step can show actual waveform bars instead of a plain bar.
async function computeWaveform(url, barCount = 80) {
  const res = await fetch(url);
  const arrayBuffer = await res.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioCtx();
  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    const rawData = audioBuffer.getChannelData(0);
    const blockSize = Math.max(1, Math.floor(rawData.length / barCount));
    const bars = [];
    for (let i = 0; i < barCount; i++) {
      let sum = 0;
      const start = i * blockSize;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(rawData[start + j] || 0);
      }
      bars.push(sum / blockSize);
    }
    const max = Math.max(...bars) || 1;
    return bars.map((b) => b / max);
  } finally {
    audioCtx.close();
  }
}

function primeDuration(audio, onReady) {
  const check = () => {
    if (isFinite(audio.duration) && audio.duration > 0) {
      onReady(audio.duration);
      return;
    }
    audio.__priming = true;
    const onTimeUpdate = () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.currentTime = 0;
      audio.__priming = false;
      onReady(isFinite(audio.duration) ? audio.duration : null);
    };
    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.currentTime = 1e101;
  };
  if (audio.readyState >= 1) {
    check();
  } else {
    audio.addEventListener("loadedmetadata", check, { once: true });
  }
}

// Browsers only allow audio.play() to succeed when called synchronously
// inside a user gesture (a click handler). This calls play() immediately,
// then seeks to `time` — instantly if duration is already known, otherwise
// via the priming workaround — without ever blocking play() on the seek.
function playFrom(audio, time) {
  const playPromise = audio.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch((err) => console.error("audio play() failed:", err));
  }
  if (isFinite(audio.duration) && audio.duration > 0) {
    audio.currentTime = time;
  } else {
    audio.__priming = true;
    const onTimeUpdate = () => {
      audio.removeEventListener("timeupdate", onTimeUpdate);
      audio.currentTime = time;
      audio.__priming = false;
    };
    audio.addEventListener("timeupdate", onTimeUpdate);
    audio.currentTime = 1e101;
  }
}

function WalkieApp({ username, userId, avatarUrl: initialAvatarUrl }) {
  // internally, "you" is still the sentinel used for all the "is this me?"
  // comparisons — displayName() below is the only thing that swaps in the
  // real chosen username wherever it's actually shown on screen.
  const realUsername = username;
  const displayName = (u) => (u === ME ? realUsername : u);
  const [myAvatarUrl, setMyAvatarUrl] = useState(initialAvatarUrl);
  const [avatarUploading, setAvatarUploading] = useState(false);
  const [showAvatarViewer, setShowAvatarViewer] = useState(false);
  const profileAvatarInputRef = useRef(null);

  const handleProfileAvatarSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAvatarUploading(true);
    try {
      const newUrl = await uploadAvatarToStorage(file, userId);
      const { error } = await supabase.from("profiles").update({ avatar_url: newUrl }).eq("id", userId);
      if (error) throw error;

      setMyAvatarUrl(newUrl);
      // patch already-loaded posts/replies of mine so the new picture shows
      // immediately everywhere, without needing a refresh
      const patchOwn = (list) =>
        list.map((p) => ({
          ...p,
          avatarUrl: p.user === ME ? newUrl : p.avatarUrl,
          replies: (p.replies || []).map((r) => (r.user === ME ? { ...r, avatarUrl: newUrl } : r)),
        }));
      setPosts(patchOwn);
      setMyPosts(patchOwn);
    } catch (err) {
      console.error("failed to update profile picture:", err);
    } finally {
      setAvatarUploading(false);
      e.target.value = "";
    }
  };

  useEffect(() => {
    if (document.getElementById("walkie-slab-font")) return;
    const link = document.createElement("link");
    link.id = "walkie-slab-font";
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@700&display=swap";
    document.head.appendChild(link);
  }, []);

  const [posts, setPosts] = useState([]);
  const [myPosts, setMyPosts] = useState([]);
  const [feedLoading, setFeedLoading] = useState(true);
  const [showWelcome, setShowWelcome] = useState(() => {
    try {
      return !localStorage.getItem("walkie_welcome_seen_v1");
    } catch {
      return false;
    }
  });
  const dismissWelcome = () => {
    setShowWelcome(false);
    try {
      localStorage.setItem("walkie_welcome_seen_v1", "1");
    } catch {}
  };
  const [feedError, setFeedError] = useState(null);

  const shapeRow = (row, ownUsername) => {
    const joinedProfile = Array.isArray(row.profiles) ? row.profiles[0] : row.profiles;
    return {
      id: row.id,
      user: row.user_id === userId ? ME : joinedProfile?.username || "unknown",
      avatarUrl: row.user_id === userId ? myAvatarUrl : joinedProfile?.avatar_url || null,
      duration: row.duration,
      caption: row.caption ?? undefined,
      timestamp: timeAgo(row.created_at),
      audioUrl: getPublicAudioUrl(row.audio_path),
      audioOffset: row.audio_offset || 0,
      heard: row.heard,
      replies: [],
    };
  };

  const POSTS_PAGE_SIZE = 20;
  const [hasMorePosts, setHasMorePosts] = useState(true);
  const [loadingMorePosts, setLoadingMorePosts] = useState(false);
  const postsOffsetRef = useRef(0);
  const repliesByPostRef = useRef({});

  const loadFeed = async () => {
    setFeedLoading(true);
    setFeedError(null);
    try {
      const [postsRes, repliesRes] = await Promise.all([
        supabase
          .from("posts")
          .select("*, profiles(username, avatar_url)")
          .order("created_at", { ascending: false })
          .range(0, POSTS_PAGE_SIZE - 1),
        // replies are already scoped by RLS to just your own posts' replies
        // and replies you've sent — a naturally small, per-user set — so
        // these aren't paginated for now, only the open-ended posts feed is.
        supabase.from("replies").select("*, profiles(username, avatar_url)").order("created_at", { ascending: true }),
      ]);
      if (postsRes.error) throw postsRes.error;
      if (repliesRes.error) throw repliesRes.error;

      const repliesByPost = {};
      for (const row of repliesRes.data) {
        const reply = shapeRow(row);
        delete reply.caption;
        delete reply.replies;
        (repliesByPost[row.post_id] ||= []).push(reply);
      }
      repliesByPostRef.current = repliesByPost;

      const shapedPosts = postsRes.data.map((row) => shapeRow(row));
      const withReplies = shapedPosts.map((p) => ({ ...p, replies: repliesByPost[p.id] || [] }));

      setPosts(withReplies);
      setMyPosts(withReplies.filter((p) => p.user === ME));
      postsOffsetRef.current = postsRes.data.length;
      setHasMorePosts(postsRes.data.length === POSTS_PAGE_SIZE);
    } catch (err) {
      console.error("failed to load feed:", err);
      setFeedError("couldn't load the feed — check your connection and try refreshing");
    } finally {
      setFeedLoading(false);
    }
  };

  const loadMorePosts = async () => {
    if (loadingMorePosts || !hasMorePosts) return;
    setLoadingMorePosts(true);
    try {
      const from = postsOffsetRef.current;
      const { data, error } = await supabase
        .from("posts")
        .select("*, profiles(username, avatar_url)")
        .order("created_at", { ascending: false })
        .range(from, from + POSTS_PAGE_SIZE - 1);
      if (error) throw error;

      const shaped = data.map((row) => shapeRow(row));
      const withReplies = shaped.map((p) => ({ ...p, replies: repliesByPostRef.current[p.id] || [] }));

      setPosts((prev) => [...prev, ...withReplies]);
      setMyPosts((prev) => [...prev, ...withReplies.filter((p) => p.user === ME)]);
      postsOffsetRef.current = from + data.length;
      setHasMorePosts(data.length === POSTS_PAGE_SIZE);
    } catch (err) {
      console.error("failed to load more posts:", err);
    } finally {
      setLoadingMorePosts(false);
    }
  };

  useEffect(() => {
    loadFeed();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // infinite scroll: load the next page automatically as you near the
  // bottom of the feed, instead of loading everything up front
  const activeScrollRef = useRef(null);
  const loadMorePostsRef = useRef(loadMorePosts);
  loadMorePostsRef.current = loadMorePosts;

  const [expandedReplies, setExpandedReplies] = useState(null);
  const [playingId, setPlayingId] = useState(null);
  const [progress, setProgress] = useState(0); // seconds elapsed on current post
  const [mixtape, setMixtape] = useState(false);
  const [mixtapeQueue, setMixtapeQueue] = useState([]);
  const [mixtapeCurrentId, setMixtapeCurrentId] = useState(null);
  const [view, setView] = useState("feed"); // feed | profile | userProfile

  const handleFeedScroll = (e) => {
    const el = e.currentTarget;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 600) {
      loadMorePostsRef.current();
    }
  };

  const [modalMode, setModalMode] = useState(null); // null | record | reply
  const [viewedUser, setViewedUser] = useState(null);
  const viewedUserAvatarUrl = posts.find((p) => p.user === viewedUser)?.avatarUrl || null;
  const [replyTarget, setReplyTarget] = useState(null);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [posted, setPosted] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedFileUrl, setUploadedFileUrl] = useState(null);
  const [uploadedDuration, setUploadedDuration] = useState(null);

  const [composeStep, setComposeStep] = useState("capture"); // capture | trim | caption
  const [clipDuration, setClipDuration] = useState(0);
  const [trimLeft, setTrimLeft] = useState(0);
  const [trimRight, setTrimRight] = useState(0);
  const [playhead, setPlayhead] = useState(0);
  const [activeDragHandle, setActiveDragHandle] = useState(null); // 'left' | 'right' | 'playhead' | null
  const [reviewPlaying, setReviewPlaying] = useState(false);
  const [captionText, setCaptionText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [captionPlayhead, setCaptionPlayhead] = useState(0);
  const [captionPlaying, setCaptionPlaying] = useState(false);
  const [micError, setMicError] = useState(null);
  const [convertingVideo, setConvertingVideo] = useState(false);
  const [showFinishRecordingConfirm, setShowFinishRecordingConfirm] = useState(false);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [deleteReplyConfirmId, setDeleteReplyConfirmId] = useState(null);
  const [editingPostId, setEditingPostId] = useState(null);
  const [editCaptionText, setEditCaptionText] = useState("");
  const [savingCaption, setSavingCaption] = useState(false);
  const [waveform, setWaveform] = useState([]);
  const [waveformLoading, setWaveformLoading] = useState(false);

  const intervalRef = useRef(null);
  const recordIntervalRef = useRef(null);
  const captionWrapRef = useRef(null);
  const captionTextRef = useRef(null);
  const loadedIdRef = useRef(null);
  const trimDragRef = useRef(null); // 'left' | 'right' | 'playhead' | 'window' | null
  const trimTrackRef = useRef(null);
  const trimLeftRef = useRef(0);
  const trimRightRef = useRef(0);
  const playheadRef = useRef(0);
  const clipDurationRef = useRef(0);
  const mainAudioRef = useRef(null);
  const audioMetaRef = useRef({ offset: 0, duration: 0 });
  const composeAudioRef = useRef(null);
  const windowDragRef = useRef({ startValue: 0, startLeft: 0, startRight: 0, startPlayhead: 0 });
  const resumeAfterScrubRef = useRef(false);
  const justStartedPlayingRef = useRef(false);
  const uploadedFileRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const hasPausedRecordingRef = useRef(false);
  const micStreamRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const recordSecondsRef = useRef(0);
  useEffect(() => {
    recordSecondsRef.current = recordSeconds;
  }, [recordSeconds]);
  useEffect(() => {
    uploadedFileRef.current = uploadedFile;
  }, [uploadedFile]);
  const [captionAnim, setCaptionAnim] = useState({ offset: 0, duration: 0, transition: false });
  const [showPlayer, setShowPlayer] = useState(false);

  const allPosts = [
    ...posts,
    ...myPosts,
    ...posts.flatMap((p) => p.replies || []),
    ...myPosts.flatMap((p) => p.replies || []),
  ];

  const hasUnheardReplies = myPosts.some((p) => (p.replies || []).some((r) => r.heard === false));

  // collapse expanded replies whenever you navigate to a different screen,
  // instead of leaving them open in the background
  useEffect(() => {
    setExpandedReplies((prev) => {
      if (prev === null) return null;
      const expandedPost = allPosts.find((p) => p.id === prev);
      if (expandedPost && (expandedPost.replies || []).some((r) => r.id === playingId)) {
        stopPlayback();
      }
      return null;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view]);

  const currentPost = mixtape
    ? allPosts.find((p) => p.id === mixtapeCurrentId) || null
    : playingId
    ? allPosts.find((p) => p.id === playingId)
    : null;

  const stopPlayback = useCallback(() => {
    clearInterval(intervalRef.current);
    if (mainAudioRef.current) mainAudioRef.current.pause();
    setPlayingId(null);
    setProgress(0);
    loadedIdRef.current = null;
  }, []);

  const pausePlayback = useCallback(() => {
    clearInterval(intervalRef.current);
    if (mainAudioRef.current) mainAudioRef.current.pause();
    setPlayingId(null);
  }, []);

  const loadAndPlay = (post) => {
    if (!post?.audioUrl || !mainAudioRef.current) return;
    const audio = mainAudioRef.current;
    const offset = post.audioOffset || 0;
    audioMetaRef.current = { offset, duration: post.duration };

    if ("mediaSession" in navigator) {
      navigator.mediaSession.metadata = new MediaMetadata({
        title: post.caption || "voice post",
        artist: displayName(post.user),
        album: "walkie",
      });
    }

    // Reassigning .src even to the same value forces a reload, so only do it
    // when the resource is actually changing.
    if (audio.src !== post.audioUrl) {
      audio.src = post.audioUrl;
    }

    const playPromise = audio.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch((err) => console.error("audio play() failed:", err));
    }

    // A single readiness event hasn't proven reliable for seeking a freshly
    // loaded remote file — the seek can silently get dropped. So instead of
    // trusting one event, keep re-applying it across the first second or so
    // of playback signals until it actually lands.
    const targetUrl = post.audioUrl;
    const events = ["loadedmetadata", "canplay", "playing", "timeupdate"];
    let attempts = 0;
    const trySeek = () => {
      if (audio.src !== targetUrl) {
        cleanup();
        return;
      }
      attempts += 1;
      if (Math.abs(audio.currentTime - offset) > 0.35) {
        audio.currentTime = offset;
      } else {
        cleanup();
      }
      if (attempts >= 6) cleanup();
    };
    const cleanup = () => events.forEach((ev) => audio.removeEventListener(ev, trySeek));
    events.forEach((ev) => audio.addEventListener(ev, trySeek));
    trySeek();
    setTimeout(cleanup, 1500);
  };

  const advance = useCallback(() => {
    if (!mixtape) {
      stopPlayback();
      return;
    }
    const idx = mixtapeQueue.findIndex((p) => p.id === mixtapeCurrentId);
    const next = mixtapeQueue[idx + 1];
    if (next) {
      setPlayingId(next.id);
      setMixtapeCurrentId(next.id);
      setProgress(0);
      loadedIdRef.current = next.id;
      if (next.audioUrl) loadAndPlay(next);
    } else {
      stopPlayback();
      setMixtape(false);
      setMixtapeCurrentId(null);
    }
  }, [mixtape, mixtapeQueue, mixtapeCurrentId, stopPlayback]);

  const goBack = useCallback(() => {
    if (!mixtape) return;
    const idx = mixtapeQueue.findIndex((p) => p.id === mixtapeCurrentId);
    const prev = mixtapeQueue[idx - 1];
    if (prev) {
      setPlayingId(prev.id);
      setMixtapeCurrentId(prev.id);
      setProgress(0);
      loadedIdRef.current = prev.id;
      if (prev.audioUrl) loadAndPlay(prev);
    }
  }, [mixtape, mixtapeQueue, mixtapeCurrentId]);

  const advanceRef = useRef(advance);
  advanceRef.current = advance;

  // simulated timer — only drives progress for posts that have no real audio
  useEffect(() => {
    if (playingId === null) return;
    const post = allPosts.find((x) => x.id === playingId);
    if (post?.audioUrl) return;
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      setProgress((p) => {
        const post = allPosts.find((x) => x.id === playingId);
        if (!post) return p;
        if (p + 0.1 >= post.duration) {
          advanceRef.current();
          return 0;
        }
        return p + 0.1;
      });
    }, 100);
    return () => clearInterval(intervalRef.current);
  }, [playingId, posts, myPosts]);

  // real audio progress sync + end-of-clip handling for posts with audioUrl —
  // wired directly as JSX props on the <audio> element below, not via a
  // ref-based effect (which wasn't reliably attaching to the live element).
  const handleMainTimeUpdate = () => {
    const audio = mainAudioRef.current;
    if (!audio || audio.__priming) return;
    const meta = audioMetaRef.current;
    const rel = audio.currentTime - meta.offset;

    if (rel >= meta.duration - 0.05) {
      audio.pause();
      setProgress(meta.duration);
      advanceRef.current();
    } else {
      setProgress(Math.max(0, rel));
    }
  };

  // safety net: if the browser reaches the real end of the file before our
  // duration-based check above catches it, this still resets playback state.
  const handleMainEnded = () => {
    setProgress(audioMetaRef.current.duration);
    advanceRef.current();
  };

  useEffect(() => {
    let cancelled = false;
    let timers = [];
    setCaptionAnim({ offset: 0, duration: 0, transition: false });

    if (!mixtape || !currentPost?.caption) return;

    const raf = requestAnimationFrame(() => {
      const wrap = captionWrapRef.current;
      const text = captionTextRef.current;
      if (!wrap || !text) return;
      const overflow = text.scrollWidth - wrap.clientWidth;
      if (overflow <= 0) return;

      const speed = 40; // px per second
      const scrollMs = Math.max(1500, (overflow / speed) * 1000);

      const cycle = () => {
        if (cancelled) return;
        setCaptionAnim({ offset: 0, duration: 0, transition: false });
        timers.push(
          setTimeout(() => {
            if (cancelled) return;
            setCaptionAnim({ offset: -overflow, duration: scrollMs, transition: true });
            timers.push(setTimeout(() => !cancelled && timers.push(setTimeout(cycle, 2000)), scrollMs));
          }, 2000)
        );
      };
      cycle();
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      timers.forEach(clearTimeout);
    };
  }, [mixtape, currentPost?.id, currentPost?.caption]);

  useEffect(() => {
    if (!mixtape) setShowPlayer(false);
  }, [mixtape]);

  // keep refs in sync for use inside the drag handler (avoids stale closures)
  useEffect(() => {
    trimLeftRef.current = trimLeft;
  }, [trimLeft]);
  useEffect(() => {
    trimRightRef.current = trimRight;
  }, [trimRight]);
  useEffect(() => {
    playheadRef.current = playhead;
  }, [playhead]);
  useEffect(() => {
    clipDurationRef.current = clipDuration;
  }, [clipDuration]);

  // global pointer handlers for dragging the trim brackets, the window
  // between them, and the playhead
  useEffect(() => {
    const getClientX = (e) => (e.touches ? e.touches[0].clientX : e.clientX);
    const getValue = (e) => {
      const rect = trimTrackRef.current.getBoundingClientRect();
      let ratio = (getClientX(e) - rect.left) / rect.width;
      ratio = Math.max(0, Math.min(1, ratio));
      return ratio * clipDurationRef.current;
    };
    const move = (e) => {
      if (!trimDragRef.current || !trimTrackRef.current) return;
      if (e.cancelable) e.preventDefault();
      const value = getValue(e);
      const minGap = 1;
      const syncAudio = (t) => {
        if (uploadedFileRef.current && composeAudioRef.current && isFinite(composeAudioRef.current.duration)) {
          composeAudioRef.current.currentTime = t;
        }
      };
      if (trimDragRef.current === "left") {
        const minLeft = Math.max(0, trimRightRef.current - MAX_RECORD_SECONDS);
        const newLeft = Math.max(minLeft, Math.min(value, trimRightRef.current - minGap));
        setTrimLeft(newLeft);
        const newPlayhead = Math.max(newLeft, Math.min(playheadRef.current, trimRightRef.current));
        setPlayhead(newPlayhead);
        syncAudio(newPlayhead);
      } else if (trimDragRef.current === "right") {
        const maxRight = Math.min(clipDurationRef.current, trimLeftRef.current + MAX_RECORD_SECONDS);
        const newRight = Math.min(maxRight, Math.max(value, trimLeftRef.current + minGap));
        setTrimRight(newRight);
        const newPlayhead = Math.min(newRight, Math.max(playheadRef.current, trimLeftRef.current));
        setPlayhead(newPlayhead);
        syncAudio(newPlayhead);
      } else if (trimDragRef.current === "window") {
        const delta = value - windowDragRef.current.startValue;
        const span = windowDragRef.current.startRight - windowDragRef.current.startLeft;
        const newLeft = Math.max(0, Math.min(windowDragRef.current.startLeft + delta, clipDurationRef.current - span));
        const newRight = newLeft + span;
        setTrimLeft(newLeft);
        setTrimRight(newRight);
        // the playhead moves with the whole window, keeping its relative
        // position inside it, instead of snapping to either edge
        const relativeOffset = windowDragRef.current.startPlayhead - windowDragRef.current.startLeft;
        const newPlayhead = newLeft + relativeOffset;
        setPlayhead(newPlayhead);
        syncAudio(newPlayhead);
      } else if (trimDragRef.current === "playhead") {
        const clamped = Math.max(trimLeftRef.current, Math.min(value, trimRightRef.current));
        setPlayhead(clamped);
        syncAudio(clamped);
      }
    };
    const up = () => {
      const wasScrubbingPlayhead = trimDragRef.current === "playhead";
      trimDragRef.current = null;
      setActiveDragHandle(null);
      if (wasScrubbingPlayhead && resumeAfterScrubRef.current) {
        resumeAfterScrubRef.current = false;
        const audio = composeAudioRef.current;
        if (audio) {
          const playPromise = audio.play();
          if (playPromise && typeof playPromise.catch === "function") {
            playPromise.catch((err) => console.error("audio play() failed:", err));
          }
        }
      }
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("touchend", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      window.removeEventListener("touchmove", move);
      window.removeEventListener("touchend", up);
    };
  }, []);

  const startWindowDrag = (e) => {
    const rect = trimTrackRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    windowDragRef.current = {
      startValue: ratio * clipDurationRef.current,
      startLeft: trimLeftRef.current,
      startRight: trimRightRef.current,
      startPlayhead: playheadRef.current,
    };
    trimDragRef.current = "window";
    setActiveDragHandle("window");
  };

  // keep the playhead inside the brackets if they move past it
  useEffect(() => {
    setPlayhead((p) => Math.max(trimLeft, Math.min(p, trimRight)));
  }, [trimLeft, trimRight]);

  // simulated playback for the trim step (recorded clips only — real uploads
  // are driven by the compose audio element's timeupdate instead)
  useEffect(() => {
    if (!reviewPlaying || uploadedFileRef.current) return;
    const id = setInterval(() => {
      setPlayhead((p) => {
        if (p + 0.1 >= trimRight) {
          setReviewPlaying(false);
          return trimRight;
        }
        return p + 0.1;
      });
    }, 100);
    return () => clearInterval(id);
  }, [reviewPlaying, trimRight]);

  // simulated playback for the caption step (recorded clips only)
  useEffect(() => {
    if (!captionPlaying || uploadedFileRef.current) return;
    const finalDuration = trimRight - trimLeft;
    const id = setInterval(() => {
      setCaptionPlayhead((p) => {
        if (p + 0.1 >= finalDuration) {
          setCaptionPlaying(false);
          return finalDuration;
        }
        return p + 0.1;
      });
    }, 100);
    return () => clearInterval(id);
  }, [captionPlaying, trimLeft, trimRight]);

  // real audio progress sync for the trim/caption compose steps
  useEffect(() => {
    const audio = composeAudioRef.current;
    if (!audio) return;
    const onTimeUpdate = () => {
      if (!uploadedFileRef.current || audio.__priming || justStartedPlayingRef.current) return;
      const rel = audio.currentTime;
      if (composeStep === "trim") {
        if (rel >= trimRightRef.current - 0.05) {
          audio.pause();
          setReviewPlaying(false);
          setPlayhead(trimRightRef.current);
        } else {
          setPlayhead(Math.max(0, rel));
        }
      } else if (composeStep === "caption") {
        const final = trimRightRef.current - trimLeftRef.current;
        const capRel = rel - trimLeftRef.current;
        if (capRel >= final - 0.05) {
          audio.pause();
          setCaptionPlaying(false);
          setCaptionPlayhead(final);
        } else {
          setCaptionPlayhead(Math.max(0, capRel));
        }
      }
    };
    audio.addEventListener("timeupdate", onTimeUpdate);
    return () => audio.removeEventListener("timeupdate", onTimeUpdate);
  }, [composeStep]);

  // decode a real waveform for the trim step, for either a recording or an upload
  useEffect(() => {
    if (composeStep !== "trim" || !uploadedFile || !uploadedFileUrl) {
      setWaveform([]);
      return;
    }
    let cancelled = false;
    setWaveform([]);
    setWaveformLoading(true);
    computeWaveform(uploadedFileUrl)
      .then((bars) => {
        if (!cancelled) setWaveform(bars);
      })
      .catch((err) => {
        console.error("waveform decode failed:", err);
        if (!cancelled) setWaveform([]);
      })
      .finally(() => {
        if (!cancelled) setWaveformLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [composeStep, uploadedFile, uploadedFileUrl]);

  const toggleReviewPlay = () => {
    const audio = composeAudioRef.current;
    if (uploadedFile && audio) {
      if (reviewPlaying) {
        audio.pause();
        setReviewPlaying(false);
        return;
      }
      if (playingId) pausePlayback();
      const startAt = playhead >= trimRight ? trimLeft : playhead;
      setPlayhead(startAt);
      setReviewPlaying(true);
      justStartedPlayingRef.current = true;
      setTimeout(() => {
        justStartedPlayingRef.current = false;
      }, 250);
      playFrom(audio, startAt);
      return;
    }
    if (reviewPlaying) {
      setReviewPlaying(false);
      return;
    }
    if (playingId) pausePlayback();
    if (playhead >= trimRight) setPlayhead(trimLeft);
    setReviewPlaying(true);
  };

  const enterTrimStep = (dur) => {
    const right = Math.min(dur, MAX_RECORD_SECONDS);
    setClipDuration(dur);
    setTrimLeft(0);
    setTrimRight(right);
    setPlayhead(0);
    setReviewPlaying(false);
    setComposeStep("trim");
    if (composeAudioRef.current) {
      composeAudioRef.current.pause();
      if (isFinite(composeAudioRef.current.duration)) {
        composeAudioRef.current.currentTime = 0;
      }
    }
  };

  const goToTrimStep = () => {
    // manual "next" from the capture step — only reachable for recorded
    // clips now; uploads auto-advance from handleFileSelect once the real
    // duration is known.
    enterTrimStep(recordSeconds);
  };

  const goToCaptionStep = () => {
    const wasPlaying = reviewPlaying;
    const resumeAt = Math.max(0, Math.min(trimRight - trimLeft, playhead - trimLeft));
    if (!wasPlaying && composeAudioRef.current) composeAudioRef.current.pause();
    setReviewPlaying(false);
    setCaptionPlayhead(wasPlaying ? resumeAt : 0);
    setCaptionPlaying(wasPlaying);
    setComposeStep("caption");
  };

  const toggleCaptionPlay = () => {
    const finalDuration = trimRight - trimLeft;
    const audio = composeAudioRef.current;
    if (uploadedFile && audio) {
      if (captionPlaying) {
        audio.pause();
        setCaptionPlaying(false);
        return;
      }
      if (playingId) pausePlayback();
      const start = captionPlayhead >= finalDuration ? 0 : captionPlayhead;
      setCaptionPlayhead(start);
      setCaptionPlaying(true);
      justStartedPlayingRef.current = true;
      setTimeout(() => {
        justStartedPlayingRef.current = false;
      }, 250);
      playFrom(audio, trimLeft + start);
      return;
    }
    if (captionPlaying) {
      setCaptionPlaying(false);
      return;
    }
    if (playingId) pausePlayback();
    setCaptionPlayhead((p) => (p >= finalDuration ? 0 : p));
    setCaptionPlaying(true);
  };

  const skipReviewBy = (delta) => {
    const next = Math.max(trimLeft, Math.min(trimRight, playhead + delta));
    setPlayhead(next);
    const audio = composeAudioRef.current;
    if (uploadedFile && audio && isFinite(audio.duration)) {
      audio.currentTime = next;
    }
  };

  const skipCaptionBy = (delta) => {
    const finalDuration = trimRight - trimLeft;
    const next = Math.max(0, Math.min(finalDuration, captionPlayhead + delta));
    setCaptionPlayhead(next);
    const audio = composeAudioRef.current;
    if (uploadedFile && audio && isFinite(audio.duration)) {
      audio.currentTime = trimLeft + next;
    }
  };

  const seekCaptionTo = (e) => {
    const finalDuration = trimRight - trimLeft;
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const next = ratio * finalDuration;
    setCaptionPlayhead(next);
    const audio = composeAudioRef.current;
    if (uploadedFile && audio && isFinite(audio.duration)) {
      audio.currentTime = trimLeft + next;
    }
  };

  const openUserProfile = (user) => {
    if (user === ME) {
      setShowPlayer(false);
      setView("profile");
      return;
    }
    setShowPlayer(false);
    setViewedUser(user);
    setView("userProfile");
  };

  const skipBy = (delta) => {
    if (!currentPost) return;
    const next = Math.max(0, Math.min(currentPost.duration, progress + delta));
    setProgress(next);
    if (currentPost.audioUrl && mainAudioRef.current) {
      mainAudioRef.current.currentTime = (currentPost.audioOffset || 0) + next;
    }
  };

  // tap-to-scrub on a progress bar — only meaningful for whichever post is
  // actually loaded right now, since that's the only one mainAudioRef reflects
  const seekPostTo = (post, e) => {
    const isActive = playingId === post.id || loadedIdRef.current === post.id;
    if (!isActive) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const target = ratio * post.duration;
    setProgress(target);
    if (post.audioUrl && mainAudioRef.current) {
      mainAudioRef.current.currentTime = (post.audioOffset || 0) + target;
    }
  };

  const toggleReplies = (id, replies = []) => {
    setExpandedReplies((prev) => {
      if (prev === id && replies.some((r) => r.id === playingId)) {
        stopPlayback();
      }
      return prev === id ? null : id;
    });
  };

  const deletePost = (id) => {
    if (playingId === id) stopPlayback();
    setMyPosts((prev) => prev.filter((p) => p.id !== id));
    setPosts((prev) => prev.filter((p) => p.id !== id));
    supabase
      .from("posts")
      .delete()
      .eq("id", id)
      .then(({ error }) => {
        if (error) console.error("failed to delete post:", error);
      });
  };

  const saveEditedCaption = async () => {
    const id = editingPostId;
    const newCaption = editCaptionText.trim();
    setSavingCaption(true);
    const { error } = await supabase.from("posts").update({ caption: newCaption }).eq("id", id);
    setSavingCaption(false);
    if (error) {
      console.error("failed to update caption:", error);
      return;
    }
    const applyEdit = (list) => list.map((p) => (p.id === id ? { ...p, caption: newCaption } : p));
    setPosts(applyEdit);
    setMyPosts(applyEdit);
    setEditingPostId(null);
  };

  const deleteReply = (replyId) => {
    if (playingId === replyId) stopPlayback();
    const stripReply = (list) =>
      list.map((p) => ({ ...p, replies: (p.replies || []).filter((r) => r.id !== replyId) }));
    setPosts(stripReply);
    setMyPosts(stripReply);
    supabase
      .from("replies")
      .delete()
      .eq("id", replyId)
      .then(({ error }) => {
        if (error) console.error("failed to delete reply:", error);
      });
  };

  const markReplyHeard = (replyId) => {
    setMyPosts((prev) =>
      prev.map((p) => ({
        ...p,
        replies: (p.replies || []).map((r) => (r.id === replyId ? { ...r, heard: true } : r)),
      }))
    );
    setPosts((prev) =>
      prev.map((p) => ({
        ...p,
        replies: (p.replies || []).map((r) => (r.id === replyId ? { ...r, heard: true } : r)),
      }))
    );
    supabase
      .from("replies")
      .update({ heard: true })
      .eq("id", replyId)
      .then(({ error }) => {
        if (error) console.error("failed to mark reply heard:", error);
      });
  };

  const handlePlay = (post) => {
    if (playingId === post.id) {
      pausePlayback();
      return;
    }
    if (reviewPlaying || captionPlaying) {
      if (composeAudioRef.current) composeAudioRef.current.pause();
      setReviewPlaying(false);
      setCaptionPlaying(false);
    }
    const audio = mainAudioRef.current;
    // A genuine pause leaves loadedIdRef pointing at this post (only a full
    // stop/finish clears it), so this only resumes in place for that case —
    // anything else gets a fresh reseek to the trim start via loadAndPlay.
    const resuming = post.audioUrl && audio && audio.src === post.audioUrl && loadedIdRef.current === post.id;
    if (!resuming) {
      setProgress(0);
    }
    if (post.heard === false) markReplyHeard(post.id);
    if (post.audioUrl && audio) {
      if (resuming) {
        const playPromise = audio.play();
        if (playPromise && typeof playPromise.catch === "function") {
          playPromise.catch((err) => console.error("audio play() failed:", err));
        }
      } else {
        loadAndPlay(post);
      }
    }
    loadedIdRef.current = post.id;
    setPlayingId(post.id);
  };

  // let the lock screen / Control Center's own play, pause, and skip buttons
  // actually control playback, and keep its play/pause icon accurate
  useEffect(() => {
    if (!("mediaSession" in navigator)) return;
    navigator.mediaSession.setActionHandler("play", () => {
      if (currentPost) handlePlay(currentPost);
    });
    navigator.mediaSession.setActionHandler("pause", () => {
      pausePlayback();
    });
    navigator.mediaSession.setActionHandler("previoustrack", mixtape ? () => goBack() : null);
    navigator.mediaSession.setActionHandler("nexttrack", mixtape ? () => advance() : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentPost, mixtape]);

  useEffect(() => {
    if (!("mediaSession" in navigator)) return;
    navigator.mediaSession.playbackState = playingId ? "playing" : "paused";
  }, [playingId]);

  const startMixtape = (list = posts) => {
    if (list.length === 0) return;
    setMixtapeQueue(list);
    setMixtape(true);
    setPlayingId(list[0].id);
    setMixtapeCurrentId(list[0].id);
    setProgress(0);
    loadedIdRef.current = list[0].id;
    if (list[0].audioUrl) loadAndPlay(list[0]);
  };

  const stopMixtape = () => {
    setMixtape(false);
    setMixtapeCurrentId(null);
    stopPlayback();
  };

  const openReply = (post) => {
    setReplyTarget(post);
    setModalMode("reply");
    setRecordSeconds(0);
    setIsRecording(false);
    setPosted(false);
    setUploadedFile(null);
    setUploadedFileUrl(null);
    setUploadedDuration(null);
    setComposeStep("capture");
    setCaptionText(""); setSubmitError(null); setSubmitting(false);
    setMicError(null);
  };

  const closeModal = () => {
    clearInterval(recordIntervalRef.current);
    if (composeAudioRef.current) composeAudioRef.current.pause();
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.onstop = null; // discard — don't advance into trim on a cancelled recording
      mediaRecorderRef.current.stop();
    }
    if (micStreamRef.current) micStreamRef.current.getTracks().forEach((t) => t.stop());
    mediaRecorderRef.current = null;
    hasPausedRecordingRef.current = false;
    setShowFinishRecordingConfirm(false);
    setModalMode(null);
    setIsRecording(false);
    setRecordSeconds(0);
    setPosted(false);
    setUploadedFile(null);
    setUploadedFileUrl(null);
    setUploadedDuration(null);
    setComposeStep("capture");
    setCaptionText(""); setSubmitError(null); setSubmitting(false);
    setReviewPlaying(false);
    setCaptionPlaying(false);
    setMicError(null);
  };

  const stopRecordingTimer = () => {
    clearInterval(recordIntervalRef.current);
  };

  const finishRealRecording = () => {
    const stream = micStreamRef.current;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    const chunks = recordedChunksRef.current;
    if (chunks.length === 0) return;
    const blob = new Blob(chunks, { type: mediaRecorderRef.current?.mimeType || "audio/webm" });
    const url = URL.createObjectURL(blob);
    setUploadedFile("voice recording");
    setUploadedFileUrl(url);
    const audio = composeAudioRef.current;
    if (audio) {
      audio.pause();
      audio.src = url;
      primeDuration(audio, (dur) => {
        setUploadedDuration(dur);
        enterTrimStep(dur && dur > 0 ? dur : recordSecondsRef.current || MAX_RECORD_SECONDS);
      });
    }
  };

  const toggleRecord = async () => {
    if (isRecording) {
      // pause, don't finalize — the session stays alive so it can resume
      stopRecordingTimer();
      setIsRecording(false);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.pause();
      }
      return;
    }

    // resuming a session we already paused — this is the actual signal that
    // the recording has more than one segment, worth confirming before finalizing
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "paused") {
      hasPausedRecordingRef.current = true;
      mediaRecorderRef.current.resume();
      setIsRecording(true);
      recordIntervalRef.current = setInterval(() => {
        setRecordSeconds((s) => {
          if (s + 1 >= MAX_RECORD_SECONDS) {
            stopRecordingTimer();
            setIsRecording(false);
            if (mediaRecorderRef.current?.state !== "inactive") mediaRecorderRef.current.stop();
            return MAX_RECORD_SECONDS;
          }
          return s + 1;
        });
      }, 1000);
      return;
    }

    // starting a brand new recording
    setMicError(null);
    setUploadedFile(null);
    setUploadedFileUrl(null);
    setUploadedDuration(null);
    hasPausedRecordingRef.current = false;

    if (navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined") {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        micStreamRef.current = stream;
        recordedChunksRef.current = [];
        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data);
        };
        recorder.onstop = finishRealRecording;
        mediaRecorderRef.current = recorder;
        recorder.start();
        setIsRecording(true);
        setRecordSeconds(0);
        recordIntervalRef.current = setInterval(() => {
          setRecordSeconds((s) => {
            if (s + 1 >= MAX_RECORD_SECONDS) {
              stopRecordingTimer();
              setIsRecording(false);
              if (mediaRecorderRef.current?.state !== "inactive") mediaRecorderRef.current.stop();
              return MAX_RECORD_SECONDS;
            }
            return s + 1;
          });
        }, 1000);
        return;
      } catch (err) {
        console.error("microphone access failed:", err);
        setMicError("Couldn't access your microphone — recording as a simulated clip instead.");
      }
    } else {
      setMicError("This browser/environment can't access the microphone — recording as a simulated clip instead.");
    }

    // fallback: no real mic access, keep the old simulated timer
    setIsRecording(true);
    recordIntervalRef.current = setInterval(() => {
      setRecordSeconds((s) => {
        if (s + 1 >= MAX_RECORD_SECONDS) {
          stopRecordingTimer();
          setIsRecording(false);
          return MAX_RECORD_SECONDS;
        }
        return s + 1;
      });
    }, 1000);
  };

  const finalizeRecordingNow = () => {
    setShowFinishRecordingConfirm(false);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop(); // onstop fires finishRealRecording -> enterTrimStep
    } else {
      goToTrimStep(); // simulated fallback path
    }
  };

  const handleNextFromRecording = () => {
    if (hasPausedRecordingRef.current) {
      setShowFinishRecordingConfirm(true);
    } else {
      finalizeRecordingNow();
    }
  };

  const discardPausedRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.onstop = null; // discard — don't advance into trim
      mediaRecorderRef.current.stop();
    }
    if (micStreamRef.current) micStreamRef.current.getTracks().forEach((t) => t.stop());
    mediaRecorderRef.current = null;
    recordedChunksRef.current = [];
    hasPausedRecordingRef.current = false;
    setRecordSeconds(0);
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    clearInterval(recordIntervalRef.current);
    setIsRecording(false);
    setMicError(null);
    setUploadedFile(file.name);
    setUploadedDuration(null);

    let workingBlob = file;

    if (file.type.startsWith("video/")) {
      setConvertingVideo(true);
      try {
        workingBlob = await extractAudioFromVideo(file);
      } catch (err) {
        console.error("failed to extract audio from video:", err);
        setConvertingVideo(false);
        setUploadedFile(null);
        setMicError("couldn't get audio out of that video on this device — try an audio file instead");
        e.target.value = "";
        return;
      }
      setConvertingVideo(false);
    }

    const url = URL.createObjectURL(workingBlob);
    setUploadedFileUrl(url);
    const audio = composeAudioRef.current;
    if (audio) {
      audio.pause();
      audio.src = url;
      primeDuration(audio, (dur) => {
        setUploadedDuration(dur);
        // enter the trim step immediately — the timeline spans the whole
        // file, with a default 2:00 window at the start to narrow from.
        enterTrimStep(dur && dur > 0 ? dur : MAX_RECORD_SECONDS);
      });
    }
  };

  const finalizeSubmit = async () => {
    clearInterval(recordIntervalRef.current);
    setIsRecording(false);
    if (composeAudioRef.current) composeAudioRef.current.pause();

    const finalDuration = Math.max(1, Math.round(trimRight - trimLeft));

    if (!uploadedFile || !uploadedFileUrl) {
      setSubmitError("no real audio to post — mic access is needed to record for real");
      return;
    }

    setSubmitting(true);
    setSubmitError(null);

    try {
      const path = await uploadAudioToStorage(uploadedFileUrl, userId);

      if (modalMode === "reply" && replyTarget) {
        const startsHeard = replyTarget.user === ME; // only true if replying to your own post
        const { data, error } = await supabase
          .from("replies")
          .insert({
            post_id: replyTarget.id,
            user_id: userId,
            duration: finalDuration,
            audio_path: path,
            audio_offset: trimLeft,
            heard: startsHeard,
          })
          .select()
          .single();
        if (error) throw error;

        const newReply = {
          id: data.id,
          user: ME,
          duration: finalDuration,
          heard: startsHeard,
          timestamp: "now",
          audioUrl: getPublicAudioUrl(path),
          audioOffset: trimLeft,
        };
        const attach = (list) =>
          list.map((p) => (p.id === replyTarget.id ? { ...p, replies: [...(p.replies || []), newReply] } : p));
        setPosts(attach);
        setMyPosts(attach);
      } else if (modalMode === "record") {
        const { data, error } = await supabase
          .from("posts")
          .insert({
            user_id: userId,
            caption: captionText.trim(),
            duration: finalDuration,
            audio_path: path,
            audio_offset: trimLeft,
          })
          .select()
          .single();
        if (error) throw error;

        const newPost = {
          id: data.id,
          user: ME,
          duration: finalDuration,
          caption: captionText.trim(),
          timestamp: "now",
          replies: [],
          audioUrl: getPublicAudioUrl(path),
          audioOffset: trimLeft,
        };
        setPosts((prev) => [newPost, ...prev]);
        setMyPosts((prev) => [newPost, ...prev]);
      }

      setSubmitting(false);
      setPosted(true);
      setTimeout(() => closeModal(), 900);
    } catch (err) {
      console.error("failed to post:", err);
      setSubmitting(false);
      setSubmitError("couldn't post — check your connection and try again");
    }
  };

  if (feedLoading) {
    return (
      <div className="min-h-screen bg-neutral-50 flex items-center justify-center">
        <p className="text-sm text-neutral-400">loading feed...</p>
      </div>
    );
  }

  if (feedError) {
    return (
      <div className="min-h-screen bg-neutral-50 flex flex-col items-center justify-center px-8 text-center gap-3">
        <p className="text-sm text-red-500">{feedError}</p>
        <button
          onClick={loadFeed}
          className="text-xs font-medium text-neutral-600 border border-neutral-300 px-4 py-2 rounded-full"
        >
          try again
        </button>
      </div>
    );
  }

  return (
    <div
      className="h-dvh bg-neutral-50 flex justify-center"
      onWheel={(e) => {
        // only forward when the wheel event originated on this outer
        // background itself (the empty desktop margins), not bubbled up
        // from the inner scrollable column — that already scrolls natively
        if (e.target === e.currentTarget && activeScrollRef.current) {
          activeScrollRef.current.scrollTop += e.deltaY;
        }
      }}
    >
      <div className="w-full max-w-sm bg-white h-dvh flex flex-col border-x border-neutral-200">
        <audio
          ref={mainAudioRef}
          className="hidden"
          onTimeUpdate={handleMainTimeUpdate}
          onEnded={handleMainEnded}
          onError={(e) => console.error("mainAudio element error:", e.currentTarget.error)}
        />
        <audio
          ref={composeAudioRef}
          className="hidden"
          onError={(e) => console.error("composeAudio element error:", e.currentTarget.error)}
        />

        {/* full-size profile picture viewer */}
        {showAvatarViewer && (
          <div
            onClick={() => setShowAvatarViewer(false)}
            className="fixed inset-0 bg-black/80 flex items-center justify-center z-30 px-6"
          >
            <button
              onClick={() => setShowAvatarViewer(false)}
              aria-label="close"
              className="absolute top-6 right-6 text-white z-10"
            >
              <X size={22} />
            </button>
            {(view === "profile" ? myAvatarUrl : viewedUserAvatarUrl) && (
              <img
                src={view === "profile" ? myAvatarUrl : viewedUserAvatarUrl}
                alt=""
                onClick={(e) => e.stopPropagation()}
                className="w-full max-w-sm aspect-square rounded-2xl object-cover"
              />
            )}
          </div>
        )}

        {/* one-time welcome popup */}
        {showWelcome && (
          <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-30 px-6">
            <div className="relative w-full max-w-xs bg-white rounded-2xl overflow-hidden">
              <button
                onClick={dismissWelcome}
                aria-label="close"
                className="absolute top-4 right-4 text-neutral-400 z-10"
              >
                <X size={18} />
              </button>
              <div className="pt-10 pb-5 px-6">
                <div className="max-h-80 overflow-y-auto pr-1">
                <div className="text-sm text-neutral-700 leading-relaxed space-y-3">
                  <p>
                    <span className="block text-lg font-semibold text-neutral-900 mb-1">hiya!</span>
                    thanks for comin on here n playin around. it's fun for me to see this idea play
                    out, and I hope you enjoy too. use walkie however you want and lemme know what you
                    notice. that said, few things to tell you:
                  </p>
                  <p>1. there's no follow/search system in here. we all share the same feed. aw! cozy.</p>
                  <p>2. replies are just between you and the person who posted, nobody else sees those.</p>
                  <p>
                    3. when you post by upload, videos won't work. :( if you wanna upload a memo
                    from iphone, you'll need to move it to files first.
                  </p>
                  <p className="italic">memo &gt; share &gt; more &gt; save to files</p>
                  <p>
                    4. finally — although listening to the mixtape while your screen is locked may work
                    for a couple tracks, it may eventually stall out. bummer! a limitation of using a web
                    app instead of a "real" one.
                  </p>
                  <p>ok! hmm... I wonder what sounds you'll share on here. 🐸</p>
                  <p>
                    thanks again,
                    <br />
                    bb
                  </p>
                </div>
              </div>
              </div>
            </div>
          </div>
        )}

        {/* header */}
        <div className="flex items-center justify-between px-5 pt-1 pb-4 border-b border-neutral-100">
          <h1
            className="text-3xl font-bold tracking-wide text-neutral-900"
            style={{ fontFamily: "'Roboto Slab', serif" }}
          >
            walkie
          </h1>

          {view === "feed" &&
            (mixtape ? (
              <button
                onClick={stopMixtape}
                className="flex items-center gap-1.5 text-xs font-medium text-white bg-neutral-900 px-3 py-1.5 rounded-full"
              >
                <Volume2 size={13} /> turn off mixtape
              </button>
            ) : (
              <button
                onClick={() => startMixtape(posts)}
                className="flex items-center gap-1.5 text-xs font-medium text-neutral-600 border border-neutral-300 px-3 py-1.5 rounded-full active:bg-neutral-100"
              >
                <Volume2 size={13} /> play mixtape
              </button>
            ))}

          {view === "userProfile" &&
            (mixtape && mixtapeQueue[0]?.user === viewedUser ? (
              <button
                onClick={stopMixtape}
                className="flex items-center gap-1.5 text-xs font-medium text-white bg-neutral-900 px-3 py-1.5 rounded-full"
              >
                <Volume2 size={13} /> turn off mixtape
              </button>
            ) : (
              <button
                onClick={() => startMixtape(posts.filter((p) => p.user === viewedUser))}
                className="flex items-center gap-1.5 text-xs font-medium text-neutral-600 border border-neutral-300 px-3 py-1.5 rounded-full active:bg-neutral-100"
              >
                <Volume2 size={13} /> play their mixtape
              </button>
            ))}

          {view === "profile" &&
            (mixtape && mixtapeQueue[0]?.user === ME ? (
              <button
                onClick={stopMixtape}
                className="flex items-center gap-1.5 text-xs font-medium text-white bg-neutral-900 px-3 py-1.5 rounded-full"
              >
                <Volume2 size={13} /> turn off mixtape
              </button>
            ) : (
              <button
                onClick={() => startMixtape(myPosts)}
                className="flex items-center gap-1.5 text-xs font-medium text-neutral-600 border border-neutral-300 px-3 py-1.5 rounded-full active:bg-neutral-100"
              >
                <Volume2 size={13} /> play your mixtape
              </button>
            ))}
        </div>

        {view === "userProfile" && (
          <button
            onClick={() => setView("feed")}
            aria-label="back"
            className="flex items-center gap-2 px-5 pt-3 pb-1"
          >
            <ArrowLeft size={18} className="text-neutral-500" />
            <span className="text-sm font-medium text-neutral-500">back to feed</span>
          </button>
        )}

        {/* feed */}
        {view === "feed" && (
        <div
          ref={activeScrollRef}
          onScroll={handleFeedScroll}
          className={`flex-1 overflow-y-auto no-scrollbar ${mixtape && currentPost ? "pb-44" : "pb-24"}`}
        >
          {posts.map((post) => {
            const isPlaying = playingId === post.id;
            const pct = (isPlaying || loadedIdRef.current === post.id) ? Math.min(100, (progress / post.duration) * 100) : 0;
            return (
              <div key={post.id} className="px-5 py-4 border-b border-neutral-100">
                <div className="flex items-center justify-between mb-2">
                  <button
                    onClick={() => openUserProfile(post.user)}
                    className="flex items-center gap-2 active:opacity-70"
                  >
                    <div className="w-8 h-8 rounded-full bg-neutral-200 flex items-center justify-center text-xs font-medium text-neutral-600 overflow-hidden">
                      {post.avatarUrl ? (
                        <img src={post.avatarUrl} alt="" className="w-8 h-8 object-cover" />
                      ) : (
                        displayName(post.user)[0].toUpperCase()
                      )}
                    </div>
                    <div className="text-left">
                      <p className="text-sm font-medium text-neutral-900">{displayName(post.user)}</p>
                      <p className="text-xs text-neutral-400">{post.timestamp}</p>
                    </div>
                  </button>
                </div>

                <p className="text-sm text-neutral-700 mb-3 break-words">{post.caption}</p>

                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handlePlay(post)}
                    className="w-10 h-10 rounded-full bg-neutral-900 text-white flex items-center justify-center flex-shrink-0 active:scale-95 transition-transform"
                    aria-label={isPlaying ? "pause" : "play"}
                  >
                    {isPlaying ? <Pause size={16} fill="white" /> : <Play size={16} fill="white" className="ml-0.5" />}
                  </button>

                  <div className="flex-1">
                    <div
                      onClick={(e) => seekPostTo(post, e)}
                      className="h-1 bg-neutral-100 rounded-full overflow-hidden cursor-pointer"
                    >
                      <div
                        className="h-full bg-neutral-900 rounded-full"
                        style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                      />
                    </div>
                    <div className="flex justify-between mt-1">
                      <span className="text-xs text-neutral-400">{(isPlaying || loadedIdRef.current === post.id) ? fmt(progress) : "0:00"}</span>
                      <span className="text-xs text-neutral-400">{fmt(post.duration)}</span>
                    </div>
                  </div>

                  {post.user !== ME && (
                    <button
                      onClick={() => openReply(post)}
                      className="text-xs font-medium text-neutral-600 border border-neutral-300 px-3 py-2 rounded-full flex-shrink-0 active:bg-neutral-100"
                    >
                      reply
                    </button>
                  )}
                </div>

                {post.user === ME ? (
                  post.replies.length > 0 && (() => {
                    const unheardCount = post.replies.filter((r) => r.heard === false).length;
                    return (
                      <>
                        <button
                          onClick={() => toggleReplies(post.id, post.replies)}
                          className={`text-xs font-medium mt-3 flex items-center gap-1 ${
                            unheardCount > 0 ? "text-red-500" : "text-neutral-500"
                          }`}
                        >
                          {expandedReplies === post.id ? (
                            <X size={14} />
                          ) : unheardCount > 0 ? (
                            "new replies"
                          ) : (
                            `${post.replies.length} ${post.replies.length === 1 ? "reply" : "replies"}`
                          )}
                        </button>

                        {expandedReplies === post.id && (
                          <div className="mt-3 pl-4 border-l border-neutral-200 space-y-3">
                            {post.replies.map((reply) => {
                              const isPlaying = playingId === reply.id;
                              const pct = (isPlaying || loadedIdRef.current === reply.id) ? Math.min(100, (progress / reply.duration) * 100) : 0;
                              return (
                                <div key={reply.id} className="flex items-center gap-3">
                                  <button
                                    onClick={() => handlePlay(reply)}
                                    className="w-8 h-8 rounded-full bg-neutral-900 text-white flex items-center justify-center flex-shrink-0 active:scale-95 transition-transform"
                                    aria-label={isPlaying ? "pause" : "play"}
                                  >
                                    {isPlaying ? (
                                      <Pause size={12} fill="white" />
                                    ) : (
                                      <Play size={12} fill="white" className="ml-0.5" />
                                    )}
                                  </button>
                                  <div className="flex-1">
                                    <p className="text-xs font-medium text-neutral-700 flex items-center gap-1.5">
                                      {displayName(reply.user)}
                                      {reply.heard === false && (
                                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" aria-label="unheard" />
                                      )}
                                    </p>
                                    <div
                                      onClick={(e) => seekPostTo(reply, e)}
                                      className="h-1 bg-neutral-100 rounded-full overflow-hidden mt-1 cursor-pointer"
                                    >
                                      <div
                                        className="h-full bg-neutral-900 rounded-full"
                                        style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                                      />
                                    </div>
                                  </div>
                                  <span className="text-xs text-neutral-400 flex-shrink-0">{reply.timestamp}</span>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </>
                    );
                  })()
                ) : (
                  post.replies.filter((r) => r.user === ME).length > 0 && (
                  <>
                    <button
                      onClick={() => toggleReplies(post.id, post.replies.filter((r) => r.user === ME))}
                      className="text-xs font-medium text-neutral-500 mt-3 flex items-center gap-1"
                    >
                      {expandedReplies === post.id ? (
                        <X size={14} />
                      ) : post.replies.filter((r) => r.user === ME).length === 1 ? (
                        "your reply"
                      ) : (
                        `your replies (${post.replies.filter((r) => r.user === ME).length})`
                      )}
                    </button>

                    {expandedReplies === post.id && (
                      <div className="mt-3 pl-4 border-l border-neutral-200 space-y-3">
                        {post.replies
                          .filter((r) => r.user === ME)
                          .map((reply) => {
                            const isPlaying = playingId === reply.id;
                            const pct = (isPlaying || loadedIdRef.current === reply.id) ? Math.min(100, (progress / reply.duration) * 100) : 0;
                            return (
                              <div key={reply.id} className="flex items-center gap-3">
                                <button
                                  onClick={() => handlePlay(reply)}
                                  className="w-8 h-8 rounded-full bg-neutral-900 text-white flex items-center justify-center flex-shrink-0 active:scale-95 transition-transform"
                                  aria-label={isPlaying ? "pause" : "play"}
                                >
                                  {isPlaying ? (
                                    <Pause size={12} fill="white" />
                                  ) : (
                                    <Play size={12} fill="white" className="ml-0.5" />
                                  )}
                                </button>
                                <div className="flex-1">
                                  <p className="text-xs font-medium text-neutral-700 mb-1">you</p>
                                  <div
                                    onClick={(e) => seekPostTo(reply, e)}
                                    className="h-1 bg-neutral-100 rounded-full overflow-hidden cursor-pointer"
                                  >
                                    <div
                                      className="h-full bg-neutral-900 rounded-full"
                                      style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                                    />
                                  </div>
                                </div>
                                <span className="text-xs text-neutral-400 flex-shrink-0">{reply.timestamp}</span>
                                <button
                                  onClick={() => setDeleteReplyConfirmId(reply.id)}
                                  aria-label="delete reply"
                                  className="text-xs font-medium text-red-500 flex-shrink-0"
                                >
                                  delete
                                </button>
                              </div>
                            );
                          })}
                      </div>
                    )}
                  </>
                  )
                )}
              </div>
            );
          })}
        </div>
        )}

        {/* single user's profile */}
        {view === "userProfile" && (
        <div ref={activeScrollRef} className={`flex-1 overflow-y-auto no-scrollbar ${mixtape && currentPost ? "pb-44" : "pb-24"}`}>
          <div className="flex items-center gap-3 px-5 py-5 border-b border-neutral-100">
            <button
              onClick={() => viewedUserAvatarUrl && setShowAvatarViewer(true)}
              aria-label="view profile picture"
              className="w-14 h-14 rounded-full bg-neutral-200 flex items-center justify-center text-base font-medium text-neutral-600 overflow-hidden flex-shrink-0 active:opacity-80"
            >
              {viewedUserAvatarUrl ? (
                <img src={viewedUserAvatarUrl} alt="" className="w-14 h-14 object-cover" />
              ) : (
                viewedUser[0]?.toUpperCase()
              )}
            </button>
            <div>
              <p className="text-2xl font-semibold text-neutral-900">{viewedUser}</p>
              <p className="text-xs text-neutral-400">
                {posts.filter((p) => p.user === viewedUser).length} posts
              </p>
            </div>
          </div>
          {posts
            .filter((p) => p.user === viewedUser)
            .map((post) => {
              const isPlaying = playingId === post.id;
              const pct = (isPlaying || loadedIdRef.current === post.id) ? Math.min(100, (progress / post.duration) * 100) : 0;
              return (
                <div key={post.id} className="px-5 py-4 border-b border-neutral-100">
                  <p className="text-xs text-neutral-400 mb-2">{post.timestamp}</p>
                  <p className="text-sm text-neutral-700 mb-3 break-words">{post.caption}</p>

                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => handlePlay(post)}
                      className="w-10 h-10 rounded-full bg-neutral-900 text-white flex items-center justify-center flex-shrink-0 active:scale-95 transition-transform"
                      aria-label={isPlaying ? "pause" : "play"}
                    >
                      {isPlaying ? <Pause size={16} fill="white" /> : <Play size={16} fill="white" className="ml-0.5" />}
                    </button>

                    <div className="flex-1">
                      <div
                        onClick={(e) => seekPostTo(post, e)}
                        className="h-1 bg-neutral-100 rounded-full overflow-hidden cursor-pointer"
                      >
                        <div
                          className="h-full bg-neutral-900 rounded-full"
                          style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                        />
                      </div>
                      <div className="flex justify-between mt-1">
                        <span className="text-xs text-neutral-400">{(isPlaying || loadedIdRef.current === post.id) ? fmt(progress) : "0:00"}</span>
                        <span className="text-xs text-neutral-400">{fmt(post.duration)}</span>
                      </div>
                    </div>

                    <button
                      onClick={() => openReply(post)}
                      className="text-xs font-medium text-neutral-600 border border-neutral-300 px-3 py-2 rounded-full flex-shrink-0 active:bg-neutral-100"
                    >
                      reply
                    </button>
                  </div>
                </div>
              );
            })}
        </div>
        )}

        {/* profile */}
        {view === "profile" && (
        <div ref={activeScrollRef} className={`flex-1 overflow-y-auto no-scrollbar ${mixtape && currentPost ? "pb-44" : "pb-24"}`}>
          <div className="flex items-center justify-between gap-3 px-5 py-5 border-b border-neutral-100">
            <div className="flex items-center gap-3">
              <div className="relative w-14 h-14 flex-shrink-0">
                <button
                  onClick={() => myAvatarUrl && setShowAvatarViewer(true)}
                  aria-label="view profile picture"
                  className="w-14 h-14 rounded-full bg-neutral-200 flex items-center justify-center text-base font-medium text-neutral-600 active:opacity-80"
                >
                  {avatarUploading ? (
                    <span className="text-[10px] text-neutral-500">...</span>
                  ) : myAvatarUrl ? (
                    <img src={myAvatarUrl} alt="" className="w-14 h-14 rounded-full object-cover" />
                  ) : (
                    realUsername[0].toUpperCase()
                  )}
                </button>
                <button
                  onClick={() => profileAvatarInputRef.current?.click()}
                  aria-label="change profile picture"
                  className="absolute bottom-0 right-0 w-5 h-5 rounded-full bg-neutral-900 flex items-center justify-center border-2 border-white active:opacity-80"
                >
                  <Camera size={9} className="text-white" />
                </button>
                <input
                  ref={profileAvatarInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleProfileAvatarSelect}
                  className="hidden"
                />
              </div>
              <div>
                <p className="text-2xl font-semibold text-neutral-900">{realUsername}</p>
                <p className="text-xs text-neutral-400">{myPosts.length} posts</p>
              </div>
            </div>
            <button
              onClick={() => setShowWelcome(true)}
              aria-label="show welcome message"
              className="w-8 h-8 rounded-full bg-neutral-100 text-neutral-400 flex items-center justify-center text-sm font-medium flex-shrink-0"
            >
              ?
            </button>
          </div>

          {myPosts.length === 0 && (
            <p className="text-sm text-neutral-400 text-center py-12">you haven't made any posts yet</p>
          )}

          {myPosts.map((post) => {
            const isPlaying = playingId === post.id;
            const pct = (isPlaying || loadedIdRef.current === post.id) ? Math.min(100, (progress / post.duration) * 100) : 0;
            return (
              <div key={post.id} className="px-5 py-4 border-b border-neutral-100">
                <div className="flex items-start justify-between gap-3 mb-3">
                  <p className="text-sm text-neutral-700 break-words flex-1">{post.caption}</p>
                  <button
                    onClick={() => {
                      setEditingPostId(post.id);
                      setEditCaptionText(post.caption || "");
                    }}
                    aria-label="edit caption"
                    className="text-xs font-medium text-neutral-500 flex-shrink-0"
                  >
                    edit
                  </button>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={() => handlePlay(post)}
                    className="w-10 h-10 rounded-full bg-neutral-900 text-white flex items-center justify-center flex-shrink-0 active:scale-95 transition-transform"
                    aria-label={isPlaying ? "pause" : "play"}
                  >
                    {isPlaying ? <Pause size={16} fill="white" /> : <Play size={16} fill="white" className="ml-0.5" />}
                  </button>

                  <div className="flex-1">
                    <div
                      onClick={(e) => seekPostTo(post, e)}
                      className="h-1 bg-neutral-100 rounded-full overflow-hidden cursor-pointer"
                    >
                      <div
                        className="h-full bg-neutral-900 rounded-full"
                        style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                      />
                    </div>
                    <div className="flex justify-between mt-1">
                      <span className="text-xs text-neutral-400">{(isPlaying || loadedIdRef.current === post.id) ? fmt(progress) : "0:00"}</span>
                      <span className="text-xs text-neutral-400">{fmt(post.duration)}</span>
                    </div>
                  </div>

                  <span className="text-xs text-neutral-400 flex-shrink-0">{post.timestamp}</span>

                  <button
                    onClick={() => setDeleteConfirmId(post.id)}
                    aria-label="delete post"
                    className="text-xs font-medium text-red-500 flex-shrink-0"
                  >
                    delete
                  </button>
                </div>

                {post.replies.length > 0 && (() => {
                  const unheardCount = post.replies.filter((r) => r.heard === false).length;
                  return (
                    <button
                      onClick={() => toggleReplies(post.id, post.replies)}
                      className={`text-xs font-medium mt-3 flex items-center gap-1 ${unheardCount > 0 ? "text-red-500" : "text-neutral-500"}`}
                    >
                      {expandedReplies === post.id ? (
                        <X size={14} />
                      ) : unheardCount > 0 ? (
                        "new replies"
                      ) : (
                        `${post.replies.length} ${post.replies.length === 1 ? "reply" : "replies"}`
                      )}
                    </button>
                  );
                })()}

                {expandedReplies === post.id && post.replies.length > 0 && (
                  <div className="mt-3 pl-4 border-l border-neutral-200 space-y-3">
                    {post.replies.map((reply) => {
                      const isPlaying = playingId === reply.id;
                      const pct = (isPlaying || loadedIdRef.current === reply.id) ? Math.min(100, (progress / reply.duration) * 100) : 0;
                      return (
                        <div key={reply.id} className="flex items-center gap-3">
                          <button
                            onClick={() => handlePlay(reply)}
                            className="w-8 h-8 rounded-full bg-neutral-900 text-white flex items-center justify-center flex-shrink-0 active:scale-95 transition-transform"
                            aria-label={isPlaying ? "pause" : "play"}
                          >
                            {isPlaying ? (
                              <Pause size={12} fill="white" />
                            ) : (
                              <Play size={12} fill="white" className="ml-0.5" />
                            )}
                          </button>
                          <div className="flex-1">
                            <p className="text-xs font-medium text-neutral-700 flex items-center gap-1.5">
                              {displayName(reply.user)}
                              {reply.heard === false && (
                                <span className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" aria-label="unheard" />
                              )}
                            </p>
                            <div
                              onClick={(e) => seekPostTo(reply, e)}
                              className="h-1 bg-neutral-100 rounded-full overflow-hidden mt-1 cursor-pointer"
                            >
                              <div
                                className="h-full bg-neutral-900 rounded-full"
                                style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                              />
                            </div>
                          </div>
                          <span className="text-xs text-neutral-400 flex-shrink-0">{reply.timestamp}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}

          {loadingMorePosts && (
            <p className="text-center text-xs text-neutral-400 py-4">loading more...</p>
          )}
        </div>
        )}

        {/* mixtape now-playing bar */}
        {mixtape && currentPost && (() => {
          const caption = currentPost.caption || "";
          const isPlaying = playingId === currentPost.id;
          return (
            <div
              onClick={() => setShowPlayer(true)}
              className="fixed bottom-20 w-full max-w-sm bg-neutral-900 text-white px-5 py-3 flex items-center justify-between cursor-pointer active:opacity-90"
            >
              <div className="min-w-0 flex items-center gap-3">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openUserProfile(currentPost.user);
                  }}
                  className="w-9 h-9 rounded-full bg-neutral-700 flex items-center justify-center text-sm font-medium flex-shrink-0 active:opacity-80 overflow-hidden"
                  aria-label={`go to ${displayName(currentPost.user)}'s profile`}
                >
                  {currentPost.avatarUrl ? (
                    <img src={currentPost.avatarUrl} alt="" className="w-9 h-9 object-cover" />
                  ) : (
                    displayName(currentPost.user)[0].toUpperCase()
                  )}
                </button>

                <div className="min-w-0">
                  <p className="text-xs text-neutral-400">now playing</p>
                  <p className="text-sm font-medium">{displayName(currentPost.user)}</p>
                  {caption && (
                    <div ref={captionWrapRef} className="overflow-hidden max-w-[170px]">
                      <p
                        ref={captionTextRef}
                        className="text-xs italic text-neutral-300 whitespace-nowrap inline-block"
                        style={{
                          transform: `translateX(${captionAnim.offset}px)`,
                          transition: captionAnim.transition ? `transform ${captionAnim.duration}ms linear` : "none",
                        }}
                      >
                        {caption}
                      </p>
                    </div>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3 flex-shrink-0">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    goBack();
                  }}
                  className="flex items-center text-xs text-neutral-300"
                  aria-label="previous track"
                >
                  <SkipBack size={16} />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePlay(currentPost);
                  }}
                  className="w-9 h-9 rounded-full bg-white text-neutral-900 flex items-center justify-center"
                  aria-label={isPlaying ? "pause" : "play"}
                >
                  {isPlaying ? (
                    <Pause size={14} fill="currentColor" />
                  ) : (
                    <Play size={14} fill="currentColor" className="ml-0.5" />
                  )}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    advance();
                  }}
                  className="flex items-center text-xs text-neutral-300"
                  aria-label="skip"
                >
                  <SkipForward size={16} />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowPlayer(true);
                  }}
                  className="flex items-center text-neutral-500"
                  aria-label="expand full screen"
                >
                  <Maximize2 size={14} />
                </button>
              </div>
            </div>
          );
        })()}

        {/* full-screen listening mode */}
        {mixtape && currentPost && showPlayer && (() => {
          const pct = Math.min(100, (progress / currentPost.duration) * 100);
          const isPlaying = playingId === currentPost.id;
          return (
            <div className="fixed inset-0 z-20 bg-neutral-900 text-white flex flex-col">
              <div className="flex justify-end px-5 pt-6">
                <button onClick={() => setShowPlayer(false)} aria-label="close">
                  <X size={22} className="text-neutral-400" />
                </button>
              </div>

              <div className="flex-1 flex flex-col items-center justify-center px-8 text-center">
                <button
                  onClick={() => openUserProfile(currentPost.user)}
                  className="w-28 h-28 rounded-full bg-neutral-700 flex items-center justify-center text-3xl font-medium mb-6 active:opacity-80 overflow-hidden"
                  aria-label={`go to ${displayName(currentPost.user)}'s profile`}
                >
                  {currentPost.avatarUrl ? (
                    <img src={currentPost.avatarUrl} alt="" className="w-28 h-28 object-cover" />
                  ) : (
                    displayName(currentPost.user)[0].toUpperCase()
                  )}
                </button>

                <button
                  onClick={() => openUserProfile(currentPost.user)}
                  className="text-lg font-semibold mb-2 active:opacity-70"
                >
                  {displayName(currentPost.user)}
                </button>

                <div className="h-16 flex items-center justify-center mb-6 max-w-xs">
                  {currentPost.caption && (
                    <p className="text-sm text-neutral-300 italic leading-relaxed break-words">
                      {currentPost.caption}
                    </p>
                  )}
                </div>

                <div className="flex items-center gap-6 mb-6">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      skipBy(-10);
                    }}
                    className="flex flex-col items-center gap-1 text-neutral-400 active:scale-90 active:text-white transition-transform"
                    aria-label="back 10 seconds"
                  >
                    <RotateCcw size={22} />
                    <span className="text-[10px]">10</span>
                  </button>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handlePlay(currentPost);
                    }}
                    className="w-16 h-16 rounded-full bg-white text-neutral-900 flex items-center justify-center"
                    aria-label={isPlaying ? "pause" : "play"}
                  >
                    {isPlaying ? (
                      <Pause size={22} fill="currentColor" />
                    ) : (
                      <Play size={22} fill="currentColor" className="ml-0.5" />
                    )}
                  </button>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      skipBy(10);
                    }}
                    className="flex flex-col items-center gap-1 text-neutral-400 active:scale-90 active:text-white transition-transform"
                    aria-label="forward 10 seconds"
                  >
                    <RotateCw size={22} />
                    <span className="text-[10px]">10</span>
                  </button>
                </div>

                <div className="w-full max-w-xs">
                  <div
                    onClick={(e) => seekPostTo(currentPost, e)}
                    className="h-1 bg-neutral-700 rounded-full overflow-hidden cursor-pointer"
                  >
                    <div
                      className="h-full bg-white rounded-full"
                      style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                    />
                  </div>
                  <div className="flex justify-between mt-1 text-xs text-neutral-400">
                    <span>{fmt(progress)}</span>
                    <span>{fmt(currentPost.duration)}</span>
                  </div>
                </div>

                <div className="mt-8 flex items-center gap-8">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      goBack();
                    }}
                    className="flex items-center gap-2 text-sm text-neutral-300"
                  >
                    <SkipBack size={18} /> last
                  </button>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      advance();
                    }}
                    className="flex items-center gap-2 text-sm text-neutral-300"
                  >
                    <SkipForward size={18} /> skip
                  </button>
                </div>
              </div>
            </div>
          );
        })()}

        {/* bottom nav */}
        <div className="fixed bottom-0 w-full max-w-sm bg-white border-t border-neutral-200 px-5 py-3 flex items-center justify-between">
          <button
            onClick={() => setView("feed")}
            className={`w-11 h-11 rounded-full flex items-center justify-center active:scale-95 transition-transform ${
              view === "feed" ? "bg-neutral-900 text-white" : "bg-neutral-200 text-neutral-500"
            }`}
            aria-label="feed"
          >
            <Radio size={18} />
          </button>

          <button
            onClick={() => {
              setReplyTarget(null);
              setModalMode("record");
              setRecordSeconds(0);
              setIsRecording(false);
              setPosted(false);
              setUploadedFile(null);
              setUploadedFileUrl(null);
              setUploadedDuration(null);
              setComposeStep("capture");
              setCaptionText(""); setSubmitError(null); setSubmitting(false);
              setMicError(null);
            }}
            className="w-14 h-14 rounded-full bg-neutral-900 text-white flex items-center justify-center active:scale-95 transition-transform"
            aria-label="new post"
          >
            <Plus size={26} />
          </button>

          <button
            onClick={() => setView("profile")}
            className={`w-11 h-11 rounded-full flex items-center justify-center active:scale-95 transition-transform ${
              view === "profile" ? "bg-neutral-900 text-white" : "bg-neutral-200 text-neutral-500"
            } ${hasUnheardReplies ? "ring-2 ring-red-500 ring-offset-2" : ""}`}
            aria-label="your profile"
          >
            <User size={18} />
          </button>
        </div>

        {/* delete confirmation */}
        {deleteConfirmId !== null && (
          <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-20 px-6">
            <div className="w-full max-w-xs bg-white rounded-2xl p-5">
              <p className="text-sm font-medium text-neutral-900 text-center">delete this post?</p>
              <p className="text-xs text-neutral-400 text-center mt-1">this can't be undone</p>
              <div className="flex gap-3 mt-5">
                <button
                  onClick={() => setDeleteConfirmId(null)}
                  className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-2.5 rounded-full"
                >
                  cancel
                </button>
                <button
                  onClick={() => {
                    deletePost(deleteConfirmId);
                    setDeleteConfirmId(null);
                  }}
                  className="flex-1 bg-red-500 text-white text-sm font-medium py-2.5 rounded-full"
                >
                  delete
                </button>
              </div>
            </div>
          </div>
        )}

        {/* finish recording confirmation */}
        {showFinishRecordingConfirm && (
          <div className="fixed inset-0 bg-black/40 flex items-end justify-center z-20 px-6 pb-32">
            <div className="w-full max-w-xs bg-white rounded-2xl p-5">
              <p className="text-sm font-medium text-neutral-900 text-center">are you finished recording?</p>
              <div className="flex gap-3 mt-5">
                <button
                  onClick={() => setShowFinishRecordingConfirm(false)}
                  className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-2.5 rounded-full"
                >
                  cancel
                </button>
                <button
                  onClick={finalizeRecordingNow}
                  className="flex-1 bg-neutral-900 text-white text-sm font-medium py-2.5 rounded-full"
                >
                  yes, finish
                </button>
              </div>
            </div>
          </div>
        )}

        {/* edit caption */}
        {editingPostId !== null && (
          <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-20 px-6">
            <div className="w-full max-w-xs bg-white rounded-2xl p-5">
              <p className="text-sm font-medium text-neutral-900 mb-3">edit caption</p>
              <textarea
                value={editCaptionText}
                onChange={(e) => {
                  if (e.target.value.length <= MAX_CAPTION_LENGTH) setEditCaptionText(e.target.value);
                }}
                maxLength={MAX_CAPTION_LENGTH}
                rows={3}
                autoFocus
                className="w-full text-sm text-neutral-900 border border-neutral-200 rounded-xl p-3 resize-none break-words focus:outline-none focus:border-neutral-400"
              />
              <p
                className={`text-xs mt-1 text-right ${
                  editCaptionText.length >= MAX_CAPTION_LENGTH ? "text-red-500" : "text-neutral-400"
                }`}
              >
                {editCaptionText.length}/{MAX_CAPTION_LENGTH}
              </p>
              <div className="flex gap-3 mt-4">
                <button
                  onClick={() => setEditingPostId(null)}
                  className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-2.5 rounded-full"
                >
                  cancel
                </button>
                <button
                  onClick={saveEditedCaption}
                  disabled={savingCaption}
                  className="flex-1 bg-neutral-900 text-white text-sm font-medium py-2.5 rounded-full disabled:opacity-50"
                >
                  {savingCaption ? "saving..." : "save"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* delete reply confirmation */}
        {deleteReplyConfirmId !== null && (
          <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-20 px-6">
            <div className="w-full max-w-xs bg-white rounded-2xl p-5">
              <p className="text-sm font-medium text-neutral-900 text-center">delete this reply?</p>
              <p className="text-xs text-neutral-400 text-center mt-1">this can't be undone</p>
              <div className="flex gap-3 mt-5">
                <button
                  onClick={() => setDeleteReplyConfirmId(null)}
                  className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-2.5 rounded-full"
                >
                  cancel
                </button>
                <button
                  onClick={() => {
                    deleteReply(deleteReplyConfirmId);
                    setDeleteReplyConfirmId(null);
                  }}
                  className="flex-1 bg-red-500 text-white text-sm font-medium py-2.5 rounded-full"
                >
                  delete
                </button>
              </div>
            </div>
          </div>
        )}

        {/* record / reply modal */}
        {modalMode !== null && (
          <div
            onClick={closeModal}
            className="fixed inset-0 bg-black/40 flex items-end justify-center z-10"
          >
            <div
              onClick={(e) => e.stopPropagation()}
              className="relative w-full max-w-sm bg-white rounded-t-2xl px-6 pt-5 pb-8"
            >
              <button onClick={closeModal} aria-label="close" className="absolute top-5 right-6 z-10">
                <X size={18} className="text-neutral-400" />
              </button>
              {modalMode === "reply" && replyTarget && (
                <div className="flex items-center gap-3 mt-6 mb-5 p-3 bg-neutral-200 rounded-full">
                  <button
                    onClick={() => handlePlay(replyTarget)}
                    className="w-9 h-9 rounded-full bg-neutral-900 text-white flex items-center justify-center flex-shrink-0"
                    aria-label={playingId === replyTarget.id ? "pause" : "play"}
                  >
                    {playingId === replyTarget.id ? (
                      <Pause size={14} fill="white" />
                    ) : (
                      <Play size={14} fill="white" className="ml-0.5" />
                    )}
                  </button>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-neutral-700 truncate">
                      {displayName(replyTarget.user)}
                    </p>
                    {replyTarget.caption && (
                      <p className="text-xs text-neutral-400 truncate mb-1">{replyTarget.caption}</p>
                    )}
                    <div
                      onClick={(e) => seekPostTo(replyTarget, e)}
                      className="h-1 bg-neutral-200 rounded-full overflow-hidden cursor-pointer"
                    >
                      <div
                        className="h-full bg-neutral-900 rounded-full"
                        style={{
                          width: `${
                            (playingId === replyTarget.id || loadedIdRef.current === replyTarget.id)
                              ? Math.min(100, (progress / replyTarget.duration) * 100)
                              : 0
                          }%`,
                          transition: playingId === replyTarget.id ? "width 0.1s linear" : "none",
                        }}
                      />
                    </div>
                  </div>
                  <span className="text-xs text-neutral-400 flex-shrink-0">{fmt(replyTarget.duration)}</span>
                </div>
              )}

              {modalMode === "reply" && (
                <div className="border-t border-neutral-100 mt-4 pt-4" />
              )}

              <div className="relative flex items-center justify-center mb-1">
                <p className="text-sm font-medium text-neutral-900">
                  {modalMode === "reply" ? "your reply" : "new post"}
                </p>
              </div>

              {modalMode === "reply" && (
                <p className="text-xs text-neutral-400 mb-2 text-center">
                  only {displayName(replyTarget?.user)} will see this
                </p>
              )}
              {modalMode === "record" && <div className="mb-6" />}

              {posted ? (
                <div className="flex flex-col items-center py-8">
                  <p className="text-sm font-medium text-neutral-900">
                    {modalMode === "reply" ? "reply sent" : "posted"}
                  </p>
                </div>
              ) : composeStep === "trim" ? (
                (() => {
                  const isPlaying = reviewPlaying;
                  const pct = clipDuration > 0 ? (playhead / clipDuration) * 100 : 0;
                  const leftPct = clipDuration > 0 ? (trimLeft / clipDuration) * 100 : 0;
                  const rightPct = clipDuration > 0 ? (trimRight / clipDuration) * 100 : 100;
                  return (
                    <div className="flex flex-col items-center py-4 w-full">
                      <div className="flex items-center gap-6 mb-6">
                        <button
                          onClick={() => skipReviewBy(-10)}
                          className="flex flex-col items-center gap-1 text-neutral-400 active:scale-90 active:text-neutral-900 transition-transform"
                          aria-label="back 10 seconds"
                        >
                          <RotateCcw size={20} />
                          <span className="text-[10px]">10</span>
                        </button>

                        <button
                          onClick={toggleReviewPlay}
                          className="w-16 h-16 rounded-full bg-neutral-900 text-white flex items-center justify-center"
                          aria-label={isPlaying ? "pause" : "play"}
                        >
                          {isPlaying ? (
                            <Pause size={20} fill="white" />
                          ) : (
                            <Play size={20} fill="white" className="ml-0.5" />
                          )}
                        </button>

                        <button
                          onClick={() => skipReviewBy(10)}
                          className="flex flex-col items-center gap-1 text-neutral-400 active:scale-90 active:text-neutral-900 transition-transform"
                          aria-label="forward 10 seconds"
                        >
                          <RotateCw size={20} />
                          <span className="text-[10px]">10</span>
                        </button>
                      </div>

                      <div
                        ref={trimTrackRef}
                        className={`relative w-full mt-2 select-none touch-none ${
                          waveform.length > 0 ? "h-16" : "h-2"
                        }`}
                      >
                        {waveform.length > 0 ? (
                          <div className="absolute inset-0 flex items-center gap-px">
                            {waveform.map((v, i) => {
                              const barTime = (i / waveform.length) * clipDuration;
                              const inWindow = barTime >= trimLeft && barTime <= trimRight;
                              return (
                                <div
                                  key={i}
                                  style={{ height: `${Math.max(8, v * 100)}%` }}
                                  className={`flex-1 rounded-full ${inWindow ? "bg-neutral-900" : "bg-neutral-300"}`}
                                />
                              );
                            })}
                          </div>
                        ) : (
                          <div className="absolute inset-0 bg-neutral-100 rounded-full" />
                        )}

                        {waveformLoading && (
                          <p className="absolute inset-x-0 -top-5 text-center text-[10px] text-neutral-400">
                            loading waveform...
                          </p>
                        )}

                        <div
                          onMouseDown={startWindowDrag}
                          onTouchStart={startWindowDrag}
                          className={`absolute top-0 h-full rounded-full cursor-grab touch-none ${
                            waveform.length > 0 ? "bg-transparent" : "bg-neutral-900"
                          }`}
                          style={{ left: `${leftPct}%`, width: `${Math.max(0, rightPct - leftPct)}%` }}
                          aria-label="selected window"
                        />

                        <div
                          onMouseDown={() => {
                            trimDragRef.current = "left";
                            setActiveDragHandle("left");
                          }}
                          onTouchStart={() => {
                            trimDragRef.current = "left";
                            setActiveDragHandle("left");
                          }}
                          className="absolute top-1/2 cursor-ew-resize touch-none"
                          style={{
                            left: `${leftPct}%`,
                            transform: "translate(-12px, -32px)",
                            padding: "0 12px",
                            paddingBottom: "28px",
                          }}
                          aria-label="trim start"
                        >
                          <div
                            className="w-1.5 h-16 border-y-4 border-l-4 border-yellow-400 transition-transform"
                            style={{ transform: `scale(${activeDragHandle === "left" ? 1.25 : 1})` }}
                          />
                        </div>
                        <div
                          onMouseDown={() => {
                            trimDragRef.current = "right";
                            setActiveDragHandle("right");
                          }}
                          onTouchStart={() => {
                            trimDragRef.current = "right";
                            setActiveDragHandle("right");
                          }}
                          className="absolute top-1/2 cursor-ew-resize touch-none"
                          style={{
                            left: `${rightPct}%`,
                            transform: "translate(-18px, -32px)",
                            padding: "0 12px",
                            paddingBottom: "28px",
                          }}
                          aria-label="trim end"
                        >
                          <div
                            className="w-1.5 h-16 border-y-4 border-r-4 border-yellow-400 transition-transform"
                            style={{ transform: `scale(${activeDragHandle === "right" ? 1.25 : 1})` }}
                          />
                        </div>
                        <div
                          onMouseDown={() => {
                            trimDragRef.current = "playhead";
                            setActiveDragHandle("playhead");
                            if (reviewPlaying && composeAudioRef.current) {
                              resumeAfterScrubRef.current = true;
                              composeAudioRef.current.pause();
                            }
                          }}
                          onTouchStart={() => {
                            trimDragRef.current = "playhead";
                            setActiveDragHandle("playhead");
                            if (reviewPlaying && composeAudioRef.current) {
                              resumeAfterScrubRef.current = true;
                              composeAudioRef.current.pause();
                            }
                          }}
                          className="absolute top-1/2 z-20 cursor-grab touch-none"
                          style={{
                            left: `${pct}%`,
                            transform: "translate(-22px, -22px)",
                            padding: "16px",
                          }}
                          aria-label="playhead"
                        >
                          <div
                            className="w-3 h-3 bg-white border-2 border-neutral-900 rounded-full shadow-sm transition-transform"
                            style={{ transform: `scale(${activeDragHandle === "playhead" ? 1.5 : 1})` }}
                          />
                        </div>
                      </div>

                      <div className="flex justify-between w-full mt-4 text-xs text-neutral-400">
                        <span>{fmt(playhead)}</span>
                        <span>
                          {fmt(trimRight - trimLeft)} selected
                          {clipDuration > trimRight - trimLeft ? ` of ${fmt(clipDuration)}` : ""}
                        </span>
                      </div>

                      {uploadedFile && clipDuration > MAX_RECORD_SECONDS && (
                        <p className="w-full text-xs text-neutral-400 mt-1">
                          drag the dark bar to move your 2:00 window, or the ends to resize it
                        </p>
                      )}

                      <div className="mt-8 w-full flex gap-3">
                        <button
                          onClick={() => {
                            if (composeAudioRef.current) composeAudioRef.current.pause();
                            setReviewPlaying(false);
                            setPlayhead(trimLeft);
                            setComposeStep("capture");
                          }}
                          className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-3 rounded-full"
                        >
                          back
                        </button>
                        <button
                          onClick={goToCaptionStep}
                          className="flex-1 bg-neutral-900 text-white text-sm font-medium py-3 rounded-full"
                        >
                          next
                        </button>
                      </div>
                    </div>
                  );
                })()
              ) : composeStep === "caption" ? (
                (() => {
                  const finalDuration = trimRight - trimLeft;
                  const isPlaying = captionPlaying;
                  const pct = finalDuration > 0 ? (captionPlayhead / finalDuration) * 100 : 0;
                  return (
                    <div className="flex flex-col items-center py-4 w-full">
                      <div className="flex items-center gap-6 mb-2">
                        <button
                          onClick={() => skipCaptionBy(-10)}
                          className="flex flex-col items-center gap-1 text-neutral-400 active:scale-90 active:text-neutral-900 transition-transform"
                          aria-label="back 10 seconds"
                        >
                          <RotateCcw size={20} />
                          <span className="text-[10px]">10</span>
                        </button>

                        <button
                          onClick={toggleCaptionPlay}
                          className="w-16 h-16 rounded-full bg-neutral-900 text-white flex items-center justify-center"
                          aria-label={isPlaying ? "pause" : "play"}
                        >
                          {isPlaying ? (
                            <Pause size={20} fill="white" />
                          ) : (
                            <Play size={20} fill="white" className="ml-0.5" />
                          )}
                        </button>

                        <button
                          onClick={() => skipCaptionBy(10)}
                          className="flex flex-col items-center gap-1 text-neutral-400 active:scale-90 active:text-neutral-900 transition-transform"
                          aria-label="forward 10 seconds"
                        >
                          <RotateCw size={20} />
                          <span className="text-[10px]">10</span>
                        </button>
                      </div>

                      <div className="w-full mt-4">
                        <div
                          onClick={seekCaptionTo}
                          className="h-2 bg-neutral-100 rounded-full overflow-hidden cursor-pointer"
                        >
                          <div
                            className="h-full bg-neutral-900 rounded-full pointer-events-none"
                            style={{ width: `${pct}%`, transition: isPlaying ? "width 0.1s linear" : "none" }}
                          />
                        </div>
                        <div className="flex justify-between mt-1 text-xs text-neutral-400">
                          <span>{fmt(captionPlayhead)}</span>
                          <span>{fmt(finalDuration)}</span>
                        </div>
                      </div>

                      {modalMode === "record" && (
                        <div className="w-full mt-6">
                          <textarea
                            value={captionText}
                            onChange={(e) => {
                              if (e.target.value.length <= MAX_CAPTION_LENGTH) setCaptionText(e.target.value);
                            }}
                            maxLength={MAX_CAPTION_LENGTH}
                            rows={3}
                            placeholder="add a caption"
                            className="w-full text-sm text-neutral-900 border border-neutral-200 rounded-xl p-3 resize-none break-words focus:outline-none focus:border-neutral-400"
                          />
                          <p
                            className={`text-xs mt-1 text-right ${
                              captionText.length >= MAX_CAPTION_LENGTH ? "text-red-500" : "text-neutral-400"
                            }`}
                          >
                            {captionText.length}/{MAX_CAPTION_LENGTH}
                          </p>
                        </div>
                      )}

                      <div className="mt-4 w-full flex gap-3">
                        <button
                          onClick={() => {
                            const wasPlaying = captionPlaying;
                            const resumePlayhead = trimLeft + captionPlayhead;
                            if (!wasPlaying && composeAudioRef.current) composeAudioRef.current.pause();
                            setCaptionPlaying(false);
                            setPlayhead(wasPlaying ? resumePlayhead : trimLeft);
                            setReviewPlaying(wasPlaying);
                            setComposeStep("trim");
                          }}
                          className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-3 rounded-full"
                        >
                          back
                        </button>
                        <button
                          onClick={finalizeSubmit}
                          disabled={submitting}
                          className="flex-1 bg-neutral-900 text-white text-sm font-medium py-3 rounded-full disabled:opacity-50"
                        >
                          {submitting ? "posting..." : modalMode === "reply" ? "send reply" : "post"}
                        </button>
                      </div>
                      {submitError && (
                        <p className="text-xs text-red-500 text-center mt-3">{submitError}</p>
                      )}
                    </div>
                  );
                })()
              ) : (
                <div className="flex flex-col items-center py-4">
                  <button
                    onClick={toggleRecord}
                    disabled={!!uploadedFile}
                    className={`w-20 h-20 rounded-full flex items-center justify-center transition-colors ${
                      isRecording ? "bg-red-500" : "bg-neutral-900"
                    } ${uploadedFile ? "opacity-30" : ""}`}
                    aria-label={isRecording ? "pause recording" : recordSeconds > 0 ? "resume recording" : "start recording"}
                  >
                    {isRecording ? (
                      <Pause size={22} className="text-white" fill="white" />
                    ) : (
                      <Mic size={26} className="text-white" />
                    )}
                  </button>

                  {uploadedFile ? (
                    <div className="mt-4 text-center">
                      <p className="text-sm text-neutral-700 truncate max-w-full">{uploadedFile}</p>
                      {convertingVideo && (
                        <p className="text-xs text-neutral-400 mt-1">extracting audio from video...</p>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-neutral-500 mt-4">
                      {fmt(recordSeconds)} / {fmt(MAX_RECORD_SECONDS)}
                    </p>
                  )}

                  {micError && (
                    <p className="text-xs text-red-500 mt-2 text-center max-w-xs">{micError}</p>
                  )}

                  <div className="w-full h-1 bg-neutral-100 rounded-full overflow-hidden mt-3">
                    <div
                      className="h-full bg-red-500 rounded-full"
                      style={{ width: uploadedFile ? "100%" : `${(recordSeconds / MAX_RECORD_SECONDS) * 100}%` }}
                    />
                  </div>

                  {!isRecording && recordSeconds === 0 && !uploadedFile && (
                    <label className="mt-6 inline-block text-center border border-neutral-300 text-neutral-700 text-sm font-medium px-6 py-3 rounded-full active:bg-neutral-100 cursor-pointer">
                      upload audio file
                      <input
                        type="file"
                        accept="audio/*,video/*,.m4a,.mp3,.wav,.aac,.caf,.aiff,.flac,.ogg"
                        onChange={handleFileSelect}
                        className="hidden"
                      />
                    </label>
                  )}

                  {uploadedFile && !convertingVideo && (
                    <div className="mt-6 w-full flex gap-3">
                      <button
                        onClick={() => {
                          setUploadedFile(null);
                          setUploadedFileUrl(null);
                          setUploadedDuration(null);
                          setRecordSeconds(0);
                          hasPausedRecordingRef.current = false;
                        }}
                        className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-3 rounded-full"
                      >
                        remove file
                      </button>
                      <button
                        onClick={() =>
                          enterTrimStep(uploadedDuration && uploadedDuration > 0 ? uploadedDuration : MAX_RECORD_SECONDS)
                        }
                        className="flex-1 bg-neutral-900 text-white text-sm font-medium py-3 rounded-full"
                      >
                        next
                      </button>
                    </div>
                  )}

                  {recordSeconds > 0 && !isRecording && !uploadedFile && (
                    <div className="mt-6 w-full flex gap-3">
                      <button
                        onClick={discardPausedRecording}
                        className="flex-1 border border-neutral-300 text-neutral-700 text-sm font-medium py-3 rounded-full"
                      >
                        back
                      </button>
                      <button
                        onClick={handleNextFromRecording}
                        className="flex-1 bg-neutral-900 text-white text-sm font-medium py-3 rounded-full"
                      >
                        next
                      </button>
                    </div>
                  )}

                  {recordSeconds >= MAX_RECORD_SECONDS && (
                    <p className="text-xs text-red-500 mt-2">time limit reached</p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// --- auth gate: shared passcode, then a real (anonymous) Supabase account
// with a chosen username, before the actual app renders ---

const APP_PASSCODE = import.meta.env.VITE_APP_PASSCODE;

// uploads a selected profile picture to the 'avatars' bucket, returning its
// public URL (not just the storage path, since profiles.avatar_url is read
// directly by <img> tags all over the app rather than resolved on the fly)
async function uploadAvatarToStorage(file, userId) {
  const ext = (file.name.match(/\.[a-zA-Z0-9]+$/)?.[0] || ".jpg").toLowerCase();
  const path = `${userId}/${Date.now()}${ext}`;
  const { error } = await supabase.storage.from("avatars").upload(path, file, {
    contentType: file.type || "image/jpeg",
  });
  if (error) throw error;
  return supabase.storage.from("avatars").getPublicUrl(path).data.publicUrl;
}

const AUTH_STEPS = ["passcode", "email", "sent"];

function AuthProgressDots({ stage }) {
  const stepIndex = AUTH_STEPS.indexOf(stage);
  if (stepIndex === -1) return null;
  return (
    <div className="flex items-center gap-2 mt-2 mb-6">
      {AUTH_STEPS.map((s, i) => (
        <div
          key={s}
          className={`w-2 h-2 rounded-full transition-colors ${
            i <= stepIndex ? "bg-neutral-900" : "bg-neutral-300"
          }`}
        />
      ))}
    </div>
  );
}

export default function Walkie() {
  const [stage, setStage] = useState("loading"); // loading | passcode | email | sent | setup | ready
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);

  const [passcodeInput, setPasscodeInput] = useState("");
  const [passcodeError, setPasscodeError] = useState("");

  const [emailInput, setEmailInput] = useState("");
  const [emailError, setEmailError] = useState("");
  const [sendingLink, setSendingLink] = useState(false);

  const [usernameInput, setUsernameInput] = useState("");
  const [usernameError, setUsernameError] = useState("");
  const [submittingUsername, setSubmittingUsername] = useState(false);
  const [avatarFile, setAvatarFile] = useState(null);
  const [avatarPreviewUrl, setAvatarPreviewUrl] = useState(null);
  const avatarInputRef = useRef(null);

  const loadProfileFor = async (sess) => {
    const { data, error } = await supabase
      .from("profiles")
      .select("*")
      .eq("id", sess.user.id)
      .maybeSingle();

    if (error) {
      console.error("failed to load profile:", error);
      setStage("setup");
      return;
    }
    if (data) {
      setProfile(data);
      setStage("ready");
    } else {
      setStage("setup");
    }
  };

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session: sess } }) => {
      if (sess) {
        setSession(sess);
        loadProfileFor(sess);
      } else {
        setStage("passcode");
      }
    });

    // catches the moment a magic link finishes signing someone in — whether
    // that resolves faster than the initial getSession() call above, or
    // happens later (the link opened in a fresh tab/session)
    const { data: listener } = supabase.auth.onAuthStateChange((_event, sess) => {
      if (sess) {
        setSession(sess);
        loadProfileFor(sess);
      }
    });
    return () => listener.subscription.unsubscribe();
  }, []);

  const submitPasscode = (e) => {
    e.preventDefault();
    setPasscodeError("");

    if (!APP_PASSCODE) {
      setPasscodeError("No passcode is configured yet (VITE_APP_PASSCODE is missing).");
      return;
    }
    if (passcodeInput !== APP_PASSCODE) {
      setPasscodeError("that's not it — try again");
      return;
    }
    setStage("email");
  };

  const submitEmail = async (e) => {
    e.preventDefault();
    setEmailError("");

    const trimmed = emailInput.trim();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmed)) {
      setEmailError("that doesn't look like a valid email");
      return;
    }

    setSendingLink(true);
    const { error } = await supabase.auth.signInWithOtp({
      email: trimmed,
      options: { emailRedirectTo: window.location.origin },
    });
    setSendingLink(false);

    if (error) {
      console.error("failed to send magic link:", error);
      setEmailError("couldn't send that link — try again in a moment");
      return;
    }
    setStage("sent");
  };

  const handleAvatarSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAvatarFile(file);
    setAvatarPreviewUrl(URL.createObjectURL(file));
  };

  const submitUsername = async (e) => {
    e.preventDefault();
    setUsernameError("");

    const trimmed = usernameInput.trim().toLowerCase();
    if (trimmed.length < 2) {
      setUsernameError("at least 2 characters");
      return;
    }
    if (!/^[a-z0-9_]+$/.test(trimmed)) {
      setUsernameError("letters, numbers, and underscores only");
      return;
    }

    setSubmittingUsername(true);
    try {
      let avatarUrl = null;
      if (avatarFile) {
        avatarUrl = await uploadAvatarToStorage(avatarFile, session.user.id);
      }
      const { data, error } = await supabase
        .from("profiles")
        .insert({ id: session.user.id, username: trimmed, avatar_url: avatarUrl })
        .select()
        .single();
      if (error) throw error;

      setProfile(data);
      setStage("ready");
    } catch (error) {
      if (error.code === "23505") {
        setUsernameError("that username's taken — try another");
      } else {
        console.error("failed to create profile:", error);
        setUsernameError("something went wrong — try again");
      }
    } finally {
      setSubmittingUsername(false);
    }
  };

  if (stage === "loading") {
    return (
      <div className="min-h-screen bg-neutral-50 flex items-center justify-center">
        <p className="text-sm text-neutral-400">loading...</p>
      </div>
    );
  }

  if (["passcode", "email", "sent"].includes(stage)) {
    return (
      <div className="min-h-screen bg-neutral-50 flex justify-center">
        <div className="w-full max-w-sm bg-white min-h-screen flex flex-col border-x border-neutral-200">
          <div className="pb-8 flex flex-col items-center" style={{ paddingTop: "28vh" }}>
            <h1
              className="text-3xl font-bold tracking-wide text-neutral-900"
              style={{ fontFamily: "'Roboto Slab', serif" }}
            >
              walkie
            </h1>
            <AuthProgressDots stage={stage} />
            <p className="text-sm text-neutral-500 text-center px-8">
              {stage === "passcode" && "enter the passcode I gave ya"}
              {stage === "email" && "enter an email to send a link to"}
              {stage === "sent" && (
                <>
                  tap the link sent to <span className="text-neutral-900">{emailInput.trim()}</span>
                </>
              )}
            </p>
          </div>

          <div className="flex flex-col items-center px-8">
            {stage === "passcode" && (
              <form onSubmit={submitPasscode} className="w-full">
                <input
                  type="password"
                  value={passcodeInput}
                  onChange={(e) => setPasscodeInput(e.target.value)}
                  placeholder="passcode"
                  autoFocus
                  className="w-full text-center text-sm border border-neutral-300 rounded-full px-4 py-3 focus:outline-none focus:border-neutral-500"
                />
                {passcodeError && (
                  <p className="text-xs text-red-500 text-center mt-2">{passcodeError}</p>
                )}
                <button
                  type="submit"
                  className="mt-4 w-full bg-neutral-900 text-white text-sm font-medium py-3 rounded-full"
                >
                  enter
                </button>
              </form>
            )}

            {stage === "email" && (
              <form onSubmit={submitEmail} className="w-full">
                <input
                  type="email"
                  value={emailInput}
                  onChange={(e) => setEmailInput(e.target.value)}
                  placeholder="you@example.com"
                  autoFocus
                  className="w-full text-center text-sm border border-neutral-300 rounded-full px-4 py-3 focus:outline-none focus:border-neutral-500"
                />
                {emailError && (
                  <p className="text-xs text-red-500 text-center mt-2">{emailError}</p>
                )}
                <button
                  type="submit"
                  disabled={sendingLink}
                  className="mt-4 w-full bg-neutral-900 text-white text-sm font-medium py-3 rounded-full disabled:opacity-50"
                >
                  {sendingLink ? "sending..." : "send link"}
                </button>
              </form>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (stage === "setup") {
    return (
      <div className="min-h-screen bg-neutral-50 flex justify-center">
        <div className="w-full max-w-sm bg-white min-h-screen flex flex-col border-x border-neutral-200">
          <div className="pb-8 flex flex-col items-center" style={{ paddingTop: "28vh" }}>
            <h1
              className="text-3xl font-bold tracking-wide text-neutral-900"
              style={{ fontFamily: "'Roboto Slab', serif" }}
            >
              walkie
            </h1>
          </div>

          <form onSubmit={submitUsername} className="w-full flex flex-col items-center px-8">
            <button
              type="button"
              onClick={() => avatarInputRef.current?.click()}
              className="relative w-24 h-24 rounded-full bg-neutral-100 flex items-center justify-center mb-6 active:opacity-80"
            >
              {avatarPreviewUrl ? (
                <img src={avatarPreviewUrl} alt="" className="w-24 h-24 rounded-full object-cover" />
              ) : (
                <User size={26} className="text-neutral-400" />
              )}
              <div className="absolute bottom-0 right-0 w-8 h-8 rounded-full bg-neutral-900 flex items-center justify-center border-2 border-white">
                <Camera size={14} className="text-white" />
              </div>
              <input
                ref={avatarInputRef}
                type="file"
                accept="image/*"
                onChange={handleAvatarSelect}
                className="hidden"
              />
            </button>

            <input
              type="text"
              value={usernameInput}
              onChange={(e) => setUsernameInput(e.target.value)}
              placeholder="pick a username"
              autoFocus
              className="w-full text-center text-sm border border-neutral-300 rounded-full px-4 py-3 focus:outline-none focus:border-neutral-500"
            />
            {usernameError && (
              <p className="text-xs text-red-500 text-center mt-2">{usernameError}</p>
            )}
            <button
              type="submit"
              disabled={submittingUsername}
              className="mt-4 w-full bg-neutral-900 text-white text-sm font-medium py-3 rounded-full disabled:opacity-50"
            >
              {submittingUsername ? "creating..." : "continue"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return <WalkieApp username={profile.username} userId={profile.id} avatarUrl={profile.avatar_url} />;
}
