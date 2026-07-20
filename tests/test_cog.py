"""Tests for the COG conversion module — CogTaskHandler and RioCogConverter."""

from unittest.mock import MagicMock, patch

import pytest

from geoembed.images.cog import (
    CogConfig,
    CogTaskHandler,
    LocalFileSystem,
    RioCogConverter,
)


class TestCogConfig:
    """Tests for CogConfig dataclass."""

    def test_defaults(self, tmp_path):
        config = CogConfig(
            input_table="cat.sch.input",
            output_table="cat.sch.output",
            dst_root=str(tmp_path / "cogs"),
        )
        assert config.compression == "deflate"
        assert config.blocksize == 512
        assert config.files_per_task == 10

    def test_frozen(self, tmp_path):
        config = CogConfig(
            input_table="cat.sch.input",
            output_table="cat.sch.output",
            dst_root=str(tmp_path / "cogs"),
        )
        with pytest.raises((AttributeError, TypeError)):
            config.compression = "lzw"  # type: ignore[misc]


class TestLocalFileSystem:
    """Tests for LocalFileSystem adapter."""

    def test_makedirs(self, tmp_path):
        fs = LocalFileSystem()
        new_dir = str(tmp_path / "nested" / "dir")
        fs.makedirs(new_dir)
        assert (tmp_path / "nested" / "dir").exists()

    def test_copy(self, tmp_path):
        fs = LocalFileSystem()
        src = tmp_path / "source.txt"
        src.write_text("hello")
        dst = str(tmp_path / "dest.txt")
        fs.copy(str(src), dst)
        assert (tmp_path / "dest.txt").read_text() == "hello"

    def test_exists(self, tmp_path):
        fs = LocalFileSystem()
        f = tmp_path / "exists.txt"
        f.write_text("x")
        assert fs.exists(str(f))
        assert not fs.exists(str(tmp_path / "nope.txt"))

    def test_remove(self, tmp_path):
        fs = LocalFileSystem()
        f = tmp_path / "to_remove.txt"
        f.write_text("x")
        fs.remove(str(f))
        assert not f.exists()

    def test_rmtree(self, tmp_path):
        fs = LocalFileSystem()
        d = tmp_path / "tree"
        d.mkdir()
        (d / "file.txt").write_text("x")
        fs.rmtree(str(d))
        assert not d.exists()


class TestCogTaskHandler:
    """Tests for CogTaskHandler with mocked converter and filesystem."""

    @patch("shutil.rmtree")
    @patch("os.makedirs")
    def test_process_file_success(self, mock_makedirs, mock_rmtree, tmp_path):
        mock_converter = MagicMock()
        mock_converter.validate.return_value = True
        mock_fs = MagicMock()

        handler = CogTaskHandler(
            converter=mock_converter,
            fs=mock_fs,
            dst_root=str(tmp_path / "output"),
        )
        result = handler.process_file("/source/image.tif")

        assert result["status"] == "success"
        assert result["error"] is None
        assert result["src"] == "/source/image.tif"
        mock_converter.convert.assert_called_once()
        mock_converter.validate.assert_called_once()
        mock_fs.copy.assert_called_once()

    @patch("shutil.rmtree")
    @patch("os.makedirs")
    def test_process_file_validation_failure(self, mock_makedirs, mock_rmtree, tmp_path):
        mock_converter = MagicMock()
        mock_converter.validate.return_value = False
        mock_fs = MagicMock()

        handler = CogTaskHandler(
            converter=mock_converter,
            fs=mock_fs,
            dst_root=str(tmp_path / "output"),
        )
        result = handler.process_file("/source/bad.tif")

        assert result["status"] == "invalid"
        assert "validation failed" in result["error"]
        mock_fs.copy.assert_not_called()

    @patch("shutil.rmtree")
    @patch("os.makedirs")
    def test_process_file_exception(self, mock_makedirs, mock_rmtree, tmp_path):
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = RuntimeError("GDAL error")
        mock_fs = MagicMock()

        handler = CogTaskHandler(
            converter=mock_converter,
            fs=mock_fs,
            dst_root=str(tmp_path / "output"),
        )
        result = handler.process_file("/source/corrupt.tif")

        assert result["status"] == "error"
        assert "GDAL error" in result["error"]

    @patch("shutil.rmtree")
    @patch("os.makedirs")
    def test_cleanup_happens_on_success(self, mock_makedirs, mock_rmtree, tmp_path):
        mock_converter = MagicMock()
        mock_converter.validate.return_value = True
        mock_fs = MagicMock()

        handler = CogTaskHandler(
            converter=mock_converter,
            fs=mock_fs,
            dst_root=str(tmp_path / "output"),
        )
        handler.process_file("/source/image.tif")

        # shutil.rmtree should be called for temp dir cleanup
        mock_rmtree.assert_called_once()
        # Verify the cleanup path is a COG temp directory
        cleanup_path = mock_rmtree.call_args[0][0]
        assert "cog_" in cleanup_path, f"Expected 'cog_' in cleanup path, got: {cleanup_path}"


class TestRioCogConverter:
    """Tests for RioCogConverter with mocked rio_cogeo."""

    def test_init_defaults(self):
        converter = RioCogConverter()
        assert converter.compression == "deflate"
        assert converter.blocksize == 512
        assert converter.overview_resampling == "nearest"

    def test_init_custom(self):
        converter = RioCogConverter(
            compression="lerc_zstd",
            blocksize=256,
            overview_resampling="average",
        )
        assert converter.compression == "lerc_zstd"
        assert converter.blocksize == 256
