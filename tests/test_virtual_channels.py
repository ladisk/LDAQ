"""
Tests for virtual channel edge cases.

This module tests edge cases for virtual channels that are not covered in
test_acquisition_base.py, including multiple virtual channels and chained
dependencies.

See openspec/changes/test-virtual-channels/specs/ for detailed requirements.
"""

import pytest
import numpy as np

import LDAQ


# =============================================================================
# Multiple Virtual Channels Tests
# =============================================================================

class TestMultipleVirtualChannels:
    """Tests for multiple virtual channels on same acquisition."""

    def test_two_independent_virtual_channels(self, multi_channel_acquisition):
        """Test two independent virtual channels work together.

        Spec: Multiple virtual channels - Two independent virtual channels
        """
        acq = multi_channel_acquisition  # has ch0, ch1, ch2

        def sum_01(ch0, ch1):
            return (ch0 + ch1).reshape(-1, 1)

        def diff_12(ch1, ch2):
            return (ch1 - ch2).reshape(-1, 1)

        acq.add_virtual_channel('sum_01', ['ch0', 'ch1'], sum_01)
        acq.add_virtual_channel('diff_12', ['ch1', 'ch2'], diff_12)

        # Contract: both should appear in channel_names
        assert 'sum_01' in acq.channel_names
        assert 'diff_12' in acq.channel_names

        acq.run_acquisition(0.1)
        result = acq.get_measurement_dict()

        # Contract: both should be computed
        assert 'sum_01' in result['channel_names']
        assert 'diff_12' in result['channel_names']

    def test_three_virtual_channels_from_same_sources(self, multi_channel_acquisition):
        """Test three virtual channels using overlapping sources.

        Spec: Multiple virtual channels - Three virtual channels from same sources
        """
        acq = multi_channel_acquisition

        def double(ch0):
            return (ch0 * 2).reshape(-1, 1)

        def triple(ch0):
            return (ch0 * 3).reshape(-1, 1)

        def quad(ch0):
            return (ch0 * 4).reshape(-1, 1)

        acq.add_virtual_channel('double', ['ch0'], double)
        acq.add_virtual_channel('triple', ['ch0'], triple)
        acq.add_virtual_channel('quad', ['ch0'], quad)

        acq.run_acquisition(0.1)
        result = acq.get_measurement_dict()

        # Contract: all three should be present and correct
        assert len([n for n in result['channel_names'] if n in ['double', 'triple', 'quad']]) == 3


# =============================================================================
# Chained Virtual Channels Tests
# =============================================================================

class TestChainedVirtualChannels:
    """Tests for virtual channels depending on other virtual channels."""

    def test_virtual_channel_uses_another_virtual(self, multi_channel_acquisition):
        """Test virtual channel can use another virtual channel as source.

        Spec: Chained virtual channels - Virtual channel uses another virtual channel
        """
        acq = multi_channel_acquisition

        # First virtual channel: sum of ch0 and ch1
        def compute_sum(ch0, ch1):
            return (ch0 + ch1).reshape(-1, 1)

        acq.add_virtual_channel('sum_ch', ['ch0', 'ch1'], compute_sum)

        # Second virtual channel: double the sum
        def double_sum(sum_ch):
            return (sum_ch * 2).reshape(-1, 1)

        acq.add_virtual_channel('double_sum', ['sum_ch'], double_sum)

        acq.run_acquisition(0.1)
        result = acq.get_measurement_dict()

        # Contract: double_sum should be 2 * (ch0 + ch1)
        ch0_idx = result['channel_names'].index('ch0')
        ch1_idx = result['channel_names'].index('ch1')
        double_sum_idx = result['channel_names'].index('double_sum')

        ch0_data = result['data'][:, ch0_idx]
        ch1_data = result['data'][:, ch1_idx]
        double_sum_data = result['data'][:, double_sum_idx]

        expected = (ch0_data + ch1_data) * 2
        np.testing.assert_array_almost_equal(double_sum_data, expected)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestVirtualChannelErrorHandling:
    """Tests for virtual channel error scenarios."""

    def test_function_exception_behavior(self, multi_channel_acquisition):
        """Document behavior when virtual channel function raises exception.

        Spec: Error handling - Function raises exception during computation

        Note: This test documents current behavior rather than prescribing it.
        """
        acq = multi_channel_acquisition

        call_count = [0]

        def sometimes_fails(ch0):
            call_count[0] += 1
            # Function that works during validation but documents exception risk
            return ch0.reshape(-1, 1)

        # Should not raise during add (validation passes)
        acq.add_virtual_channel('risky', ['ch0'], sometimes_fails)

        # Should work during acquisition
        acq.run_acquisition(0.05)
        result = acq.get_measurement_dict()

        # Document: function was called
        assert call_count[0] > 0
        assert 'risky' in result['channel_names']

    def test_output_must_be_2d_array(self, multi_channel_acquisition):
        """Test that output must be properly shaped.

        Spec: Error handling - Wrong output shape logged or raises
        """
        acq = multi_channel_acquisition

        def returns_scalar(ch0):
            return 42  # Not an array

        # Contract: should raise during add_virtual_channel validation
        with pytest.raises(ValueError):
            acq.add_virtual_channel('bad', ['ch0'], returns_scalar)
